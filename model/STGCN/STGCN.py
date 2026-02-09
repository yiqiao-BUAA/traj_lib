import torch.nn as nn
import torch
import torch.nn.functional as F


# =====================================================================
# Model: teacher-forcing encoder + per-key subgraph GCN candidates
# =====================================================================
class CausalTCN(nn.Module):
    """
    Causal TCN that outputs per-step hidden states.
    Input : x [B,S,D], mask [B,S] bool
    Output: h [B,S,D] (each step depends only on <=t)
    """
    def __init__(self, d_model: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.kernel = kernel
        self.left_pad = kernel - 1
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=0)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=0)
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _causal(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return conv(x)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x * mask.unsqueeze(-1)
        x_c = x.transpose(1, 2)                       # [B,D,S]
        y = F.relu(self._causal(self.conv1, x_c))
        y = self._causal(self.conv2, y).transpose(1, 2)  # [B,S,D]
        y = self.ln(y + x)
        y = self.drop(y)
        y = y * mask.unsqueeze(-1)
        return y


class SubgraphGCNLayer(nn.Module):
    """
    Dense GCN layer on a subgraph:
      H = ReLU( D^{-1/2}(A+I)D^{-1/2} X W )
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # X: [n_sub, D], A: [n_sub,n_sub]
        n = A.size(0)
        I = torch.eye(n, device=A.device, dtype=A.dtype)
        A_hat = A + I

        deg = A_hat.sum(-1).clamp(min=1e-6)          # [n_sub]
        inv_sqrt = deg.pow(-0.5)
        A_norm = inv_sqrt.unsqueeze(1) * A_hat * inv_sqrt.unsqueeze(0)  # [n_sub,n_sub]

        H = A_norm @ X                               # [n_sub,D]
        H = F.relu(self.lin(H))
        return H


class CandidateSubgraphTFModel(nn.Module):
    """
    Teacher forcing model producing per-step context h_t,
    and scoring candidates defined by per-key subgraph.

    Sequence encoder:
      embed(x_in) -> causal TCN -> h_seq [B,S-1,D]

    Candidate scorer:
      for each key(prev_id), build candidate set (sub_nodes), slice A_sub,
      compute candidate embeddings via small GCN, then logits = h @ cand_emb^T
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout: float,
        tcn_kernel: int = 3,
        gcn_layers: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tcn = CausalTCN(d_model=d_model, kernel=tcn_kernel, dropout=dropout)

        self.gcn1 = SubgraphGCNLayer(d_model, d_model)
        self.gcn2 = SubgraphGCNLayer(d_model, d_model) if gcn_layers >= 2 else None

        self.h_proj = nn.Linear(d_model, d_model, bias=False)

    def encode(self, x_in: torch.Tensor, mask_in: torch.Tensor) -> torch.Tensor:
        x_emb = self.embed(x_in)                  # [B,S-1,D]
        h_seq = self.tcn(x_emb, mask_in)          # [B,S-1,D]
        h_seq = self.h_proj(h_seq)                # [B,S-1,D]
        return h_seq

    def candidate_embed(self, sub_nodes: torch.Tensor, A_sub: torch.Tensor) -> torch.Tensor:
        """
        sub_nodes: [n_sub] global ids (on GPU)
        A_sub:     [n_sub,n_sub] (on GPU)
        returns cand_emb: [n_sub,D]
        """
        X = self.embed(sub_nodes)                 # [n_sub,D]
        X = self.gcn1(X, A_sub)
        if self.gcn2 is not None:
            X = self.gcn2(X, A_sub)
        return X