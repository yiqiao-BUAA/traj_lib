import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from utils.exargs import ConfigResolver

model_args = ConfigResolver("./model/CLSPRec/CLSPRec.yaml").parse()


class CheckInEmbedding(nn.Module):
    """Embedding layer for check-in features including POI, category, user, hour, and day."""

    def __init__(self, f_embed_size: int, vocab_size: dict):
        super().__init__()
        self.embed_size = f_embed_size
        self.vocab_size = vocab_size
        
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        # Create embedding layers
        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(
            user_num + 1, self.embed_size, padding_idx=user_num
        )
        self.hour_embed = nn.Embedding(
            hour_num + 1, self.embed_size, padding_idx=hour_num
        )
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape [B, 5, L] (Batch) or [5, L] (Single)
        Returns:
            torch.Tensor: Shape [B, L, 5*Embed] or [L, 5*Embed]
        """
        # Handle batch dimension: if 3D [B, 5, L], we slice on dim 1. 
        # If 2D [5, L], we slice on dim 0.
        if x.dim() == 3:
            poi_idx, cat_idx, user_idx, hour_idx, day_idx = (
                x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
            )
        else:
            poi_idx, cat_idx, user_idx, hour_idx, day_idx = (
                x[0], x[1], x[2], x[3], x[4]
            )

        poi_emb = self.poi_embed(poi_idx)
        cat_emb = self.cat_embed(cat_idx)
        user_emb = self.user_embed(user_idx)
        hour_emb = self.hour_embed(hour_idx)
        day_emb = self.day_embed(day_idx)

        # Concatenate along the feature dimension (last dimension)
        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb), dim=-1)


class SelfAttention(nn.Module):
    """Self-attention mechanism implementation with Batch support."""

    def __init__(self, embed_size: int, heads: int):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
            self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(
        self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            values, keys, query: [Batch, SeqLen, Embed]
            mask: [Batch, SeqLen] (1 for valid, 0 for pad) or None
        """
        # Get Batch size (N) and Sequence Lengths
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Linear projections
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Reshape for multi-head: [N, Seq, Heads, HeadDim]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Compute energy: [N, Heads, QueryLen, KeyLen]
        # Einsum: Batch(n), Heads(h), Query(q), Key(k), Dim(d)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            # Mask shape [N, KeyLen]. We need to expand to [N, 1, 1, KeyLen]
            # so it broadcasts over Heads and QueryLen
            expanded_mask = mask.unsqueeze(1).unsqueeze(2) 
            # Apply mask: fill 0 locations (padding) with -inf
            energy = energy.masked_fill(expanded_mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Apply attention to values: [N, Heads, QueryLen, KeyLen] * [N, KeyLen, Heads, Dim] 
        # -> [N, QueryLen, Heads, Dim]
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class EncoderBlock(nn.Module):
    """Transformer encoder block with batch masking support."""

    def __init__(
        self, embed_size: int, heads: int, dropout: float, forward_expansion: int
    ):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        attention = self.attention(value, key, query, mask=mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerEncoder(nn.Module):
    """Transformer encoder consisting of multiple encoder blocks."""

    def __init__(
        self,
        embedding_layer: nn.Module,
        embed_size: int,
        num_encoder_layers: int,
        num_heads: int,
        forward_expansion: int,
        dropout: float,
    ):
        super(TransformerEncoder, self).__init__()
        self.embedding_layer = embedding_layer
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            feature_seq: [Batch, 5, SeqLen]
            mask: [Batch, SeqLen]
        """
        embedding = self.embedding_layer(feature_seq)  # [Batch, SeqLen, Dim]
        out = self.dropout(embedding)

        for layer in self.layers:
            out = layer(out, out, out, mask=mask)

        return out


class Attention(nn.Module):
    """Attention mechanism for queries and keys with different dimensions (Batch supported)."""

    def __init__(self, qdim: int, kdim: int):
        super().__init__()
        self.expansion = nn.Linear(qdim, kdim)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query: [Batch, Q_Dim]
            key: [Batch, Seq_Len, K_Dim]
            value: [Batch, Seq_Len, K_Dim]
            mask: [Batch, Seq_Len]
        """
        q = self.expansion(query)  # [Batch, K_Dim]
        
        # Calculate scores: (Batch, 1, K_Dim) @ (Batch, K_Dim, Seq_Len) -> (Batch, 1, Seq_Len)
        temp = torch.bmm(q.unsqueeze(1), key.transpose(1, 2))
        
        if mask is not None:
             # Mask is [Batch, Seq_Len], expand to [Batch, 1, Seq_Len]
             temp = temp.masked_fill(mask.unsqueeze(1) == 0, float("-1e20"))

        weight = torch.softmax(temp, dim=2)  # [Batch, 1, Seq_Len]
        
        # Weighted sum: (Batch, 1, Seq_Len) @ (Batch, Seq_Len, K_Dim) -> (Batch, 1, K_Dim)
        temp2 = torch.bmm(weight, value)
        out = temp2.squeeze(1)  # [Batch, K_Dim]

        return out


class CLSPRec(nn.Module):
    """CLSPRec model for sequential recommendation with contrastive learning."""

    def __init__(
        self,
        vocab_size: dict,
        f_embed_size: int = 60,
        num_encoder_layers: int = 1,
        num_lstm_layers: int = 1,
        num_heads: int = 1,
        forward_expansion: int = 2,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.total_embed_size = f_embed_size * 5
        self.num_poi = vocab_size["POI"]
        # Layers
        self.embedding = CheckInEmbedding(f_embed_size, vocab_size)
        self.encoder = TransformerEncoder(
            self.embedding,
            self.total_embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout_p,
        )
        # Using batch_first=True to handle [Batch, Seq, Dim] standard format
        self.lstm = nn.LSTM(
            input_size=self.total_embed_size,
            hidden_size=self.total_embed_size,
            num_layers=num_lstm_layers,
            dropout=0,
            batch_first=True
        )
        self.final_attention = Attention(qdim=f_embed_size, kdim=self.total_embed_size)
        self.out_linear = nn.Sequential(
            nn.Linear(self.total_embed_size, self.total_embed_size * forward_expansion),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.total_embed_size * forward_expansion, self.num_poi),
        )

        self.loss_func = nn.CrossEntropyLoss()
        self.tryone_line2 = nn.Linear(self.total_embed_size, f_embed_size)
        self.enhance_val = nn.Parameter(torch.tensor(0.5))

    def feature_mask_batch(self, sequences: torch.Tensor, mask_prop: float, mask_len: torch.Tensor) -> torch.Tensor:
        """
        Apply random masking to input batch tensors for contrastive learning.
        sequences: [Batch, 5, MaxLen]
        mask_len: [Batch] (actual lengths)
        """
        B, _, MaxLen = sequences.shape
        masked_seqs = sequences.clone()
        
        # Generate random mask probabilities
        rand_matrix = torch.rand((B, MaxLen), device=sequences.device)
        
        # Create sequence length mask to avoid masking padding
        seq_range = torch.arange(MaxLen, device=sequences.device).unsqueeze(0).expand(B, MaxLen)
        valid_range = seq_range < mask_len.unsqueeze(1)
        
        # Determine items to mask
        to_mask = (rand_matrix < mask_prop) & valid_range
        
        # Apply masks
        masked_seqs[:, 0, :][to_mask] = self.vocab_size["POI"]
        masked_seqs[:, 1, :][to_mask] = self.vocab_size["cat"]
        masked_seqs[:, 3, :][to_mask] = self.vocab_size["hour"]
        masked_seqs[:, 4, :][to_mask] = self.vocab_size["day"]

        return masked_seqs

    def ssl_batch(
        self,
        embedding_1: torch.Tensor,
        embedding_2: torch.Tensor,
        neg_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized SSL Loss.
        Inputs: [Batch, Dim]
        """
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), dim=1) # Dot product per batch item

        pos = score(embedding_1, embedding_2)
        neg1 = score(embedding_1, neg_embedding)
        neg2 = score(embedding_2, neg_embedding)
        neg = (neg1 + neg2) / 2
        
        # Sigmoid and Log
        epsilon = 1e-8
        pos_part = -torch.log(torch.sigmoid(pos) + epsilon)
        neg_part = -torch.log(1 - torch.sigmoid(neg) + epsilon)
        
        return torch.mean(pos_part + neg_part)

    def _pad_and_create_mask(self, sample_list):
        """
        Helper to convert list of [5, Len] tensors to [Batch, 5, MaxLen] and generate mask.
        """
        # Original sample items are [5, Len].
        # Transpose to [Len, 5] for pad_sequence
        transposed_samples = [s.t() for s in sample_list]
        
        # Pad: [Batch, MaxLen, 5]
        # Pad with 0 temporarily
        padded = pad_sequence(transposed_samples, batch_first=True, padding_value=0)
        
        # Calculate lengths
        lengths = torch.tensor([s.size(0) for s in transposed_samples], device=model_args.device)
        B, MaxLen, _ = padded.shape
        
        # Create Boolean Mask [Batch, MaxLen] (True = Valid, False = Pad)
        range_tensor = torch.arange(MaxLen, device=model_args.device).unsqueeze(0).expand(B, MaxLen)
        mask = range_tensor < lengths.unsqueeze(1)
        
        # Permute to [Batch, 5, MaxLen]
        padded = padded.permute(0, 2, 1).contiguous()
        
        # Apply specific padding values to the padded areas (where mask is False)
        padded[:, 0, :][~mask] = self.vocab_size["POI"]
        padded[:, 1, :][~mask] = self.vocab_size["cat"]
        padded[:, 2, :][~mask] = self.vocab_size["user"]
        padded[:, 3, :][~mask] = self.vocab_size["hour"]
        padded[:, 4, :][~mask] = self.vocab_size["day"]
        
        return padded, mask, lengths

    def forward(
        self,
        sample: list,
        long_term_seq: list,
        label: torch.Tensor,
        neg_sample_list: list,
    ) -> tuple:
        """
        Optimized Forward pass using Batch Processing.
        """
        device = model_args.device
        # 1. Prepare Batch Tensors (Vectorization)
        # Short-term samples
        short_seqs, short_mask, _ = self._pad_and_create_mask(sample) # [B, 5, L]
        user_ids = short_seqs[:, 2, 0] # Extract user IDs

        # Negative samples
        if neg_sample_list and len(neg_sample_list) > 0:
            neg_seqs, neg_mask, _ = self._pad_and_create_mask(neg_sample_list)
        else:
            neg_seqs, neg_mask = None, None

        # Long-term samples (Handle None)
        valid_indices = [i for i, x in enumerate(long_term_seq) if x is not None]
        long_term_exists = len(valid_indices) > 0
        
        if long_term_exists:
            valid_long_samples = [long_term_seq[i] for i in valid_indices]
            long_seqs_valid, long_mask_valid, long_lens_valid = self._pad_and_create_mask(valid_long_samples)
            
            # Apply Random Masking
            long_seqs_valid = self.feature_mask_batch(long_seqs_valid, model_args.mask_prop, long_lens_valid)
            
            # Encoder Pass
            long_term_out_valid = self.encoder(long_seqs_valid, mask=long_mask_valid) # [Val_B, MaxL, Dim]
            
            # Mean pooling for SSL (masked mean)
            mask_float = long_mask_valid.float().unsqueeze(-1)
            sum_long = torch.sum(long_term_out_valid * mask_float, dim=1)
            count_long = torch.sum(mask_float, dim=1).clamp(min=1e-9)
            long_embed_mean_valid = sum_long / count_long
        
        # 2. Main Model Processing
        
        # Short-term Encoder
        short_term_state = self.encoder(short_seqs, mask=short_mask) # [B, L, Dim]
        
        # LSTM & User Enhancement
        short_emb = self.embedding(short_seqs) # [B, L, Dim]
        lstm_out, _ = self.lstm(short_emb) # [B, L, Dim]
        
        # Compute mean for User Enhancement
        mask_float_short = short_mask.float().unsqueeze(-1)
        short_term_mean = torch.sum(lstm_out * mask_float_short, dim=1) / torch.sum(mask_float_short, dim=1).clamp(min=1e-9)
        
        # User Embedding Mixing
        user_embed = self.embedding.user_embed(user_ids)
        user_embed = self.enhance_val * user_embed + (1 - self.enhance_val) * self.tryone_line2(short_term_mean)

        # 3. Reconstruct Long Term Batch
        B, L_short, Dim = short_term_state.shape
        long_term_out = torch.zeros((B, 1, Dim), device=device) # Placeholder
        
        if long_term_exists:
            max_long_len = long_term_out_valid.size(1)
            long_term_out = torch.zeros((B, max_long_len, Dim), device=device)
            long_term_out[valid_indices] = long_term_out_valid
            
            # Also need embedding mean for SSL
            long_embed_mean = torch.zeros((B, Dim), device=device)
            long_embed_mean[valid_indices] = long_embed_mean_valid
        else:
            long_term_out = torch.zeros((B, 1, Dim), device=device)
            long_embed_mean = torch.zeros((B, Dim), device=device)
            
        # 4. SSL Loss
        short_embed_mean = torch.sum(short_term_state * mask_float_short, dim=1) / torch.sum(mask_float_short, dim=1).clamp(min=1e-9)
        
        if neg_seqs is not None:
            neg_out = self.encoder(neg_seqs, mask=neg_mask)
            mask_float_neg = neg_mask.float().unsqueeze(-1)
            neg_embed_mean = torch.sum(neg_out * mask_float_neg, dim=1) / torch.sum(mask_float_neg, dim=1).clamp(min=1e-9)
            
            # --- FIX: Match batch sizes ---
            curr_bs = short_embed_mean.size(0)
            neg_bs = neg_embed_mean.size(0)
            
            if neg_bs > curr_bs:
                neg_embed_mean = neg_embed_mean[:curr_bs]
            elif neg_bs < curr_bs:
                repeats = (curr_bs // neg_bs) + 1
                neg_embed_mean = neg_embed_mean.repeat(repeats, 1)[:curr_bs]
            # ------------------------------
            
            ssl_loss = self.ssl_batch(short_embed_mean, long_embed_mean, neg_embed_mean)
        else:
            ssl_loss = torch.tensor(0.0, device=device)

        # 5. Final Prediction
        h_all = torch.cat((short_term_state, long_term_out), dim=1) # [B, Total_Len, Dim]
        
        # Construct combined mask
        if long_term_exists:
            full_long_mask = torch.zeros((B, long_term_out.size(1)), dtype=torch.bool, device=device)
            full_long_mask[valid_indices] = long_mask_valid
            final_mask = torch.cat((short_mask, full_long_mask), dim=1)
        else:
            dummy_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
            final_mask = torch.cat((short_mask, dummy_mask), dim=1)

        # Final Attention
        final_att = self.final_attention(user_embed, h_all, h_all, mask=final_mask)
        preds = self.out_linear(final_att)

        pred_loss = self.loss_func(preds, label.long())
        loss = pred_loss + ssl_loss * model_args.neg_weight
        
        return loss, preds

    def predict(
        self,
        sample: list,
        long_term_seq: list,
        label: torch.Tensor,
        neg_sample_list: list,
    ) -> tuple:
        """
        Inference method.
        """
        _, pred_raw = self.forward(sample, long_term_seq, label, neg_sample_list)
        ranking = torch.sort(pred_raw, descending=True)[1]
        return ranking, label, pred_raw