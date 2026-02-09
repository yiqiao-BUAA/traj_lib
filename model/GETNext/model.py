import math
from typing import Callable, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

def t2v(
    tau: torch.Tensor,
    f: Callable,
    out_features: int,
    w: torch.Tensor,
    b: torch.Tensor,
    w0: torch.Tensor,
    b0: torch.Tensor,
    arg: Any = None,
) -> torch.Tensor:
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, out_dim: int):
        super(Time2Vec, self).__init__()
        self.l1 = SineActivation(1, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x.view(-1, 1))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500, batch_first: bool = True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.batch_first:
            x = x + self.pe[:x.size(1), :].transpose(0, 1)
        else:
            x = x + self.pe[:x.size(0), :]
            
        return self.dropout(x)

class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = torch.cat((user_embed, poi_embed), -1)
        x = self.fuse_embed(x)
        x = self.leaky_relu(x)
        return x


class GETNextModel(nn.Module):
    def __init__(self, node_attn_in_features: int, node_attn_nhid: int, 
                 gcn_ninput: int, gcn_nhid: list, gcn_dropout: float, 
                 seq_input_embed: int, transformer_dropout: float, transformer_nhead: int, transformer_nhid: int, transformer_nlayers: int, 
                 poi_embed_dim: int, user_embed_dim: int, cat_embed_dim: int, time_embed_dim: int, 
                 num_cats: int, num_users: int,  num_pois: int, device: torch.device):
        super(GETNextModel, self).__init__()
        self.device = device
        # NodeAttnMap
        self.node_attn_model = NodeAttnMap(node_attn_in_features, node_attn_nhid)
        
        # GCN
        self.gcn = GCN(gcn_ninput, gcn_nhid, poi_embed_dim, gcn_dropout)
        
        # UserEmbedding
        self.user_emb = UserEmbeddings(num_users, user_embed_dim)
        
        # CategoryEmbedding
        self.cat_emb = CategoryEmbeddings(num_cats, cat_embed_dim)
        
        # FuseEmbedding1
        self.fuse_embed1 = FuseEmbeddings(user_embed_dim, poi_embed_dim)
        
        # FuseEmbedding2
        self.fuse_embed2 = FuseEmbeddings(cat_embed_dim, time_embed_dim)
        
        # TimeEmbedding
        self.time_embed_model = Time2Vec(time_embed_dim)
        
        # TransformerModel
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(seq_input_embed, transformer_dropout, batch_first=True)
        encoder_layers = TransformerEncoderLayer(seq_input_embed, transformer_nhead, transformer_nhid, transformer_dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, transformer_nlayers)
        self.embed_size = seq_input_embed
        self.decoder_poi = nn.Linear(seq_input_embed, num_pois)
        self.decoder_time = nn.Linear(seq_input_embed, 1)
        self.decoder_cat = nn.Linear(seq_input_embed, num_cats)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def predict(self, batch: Dict[str, Any], X: torch.Tensor, A: torch.Tensor):
        
        all_poi_embeddings = self.gcn(X, A)
        input_user_id = batch['user_id'].to(self.device)
        input_seqs = batch['POI_id'].to(self.device)
        input_seq_cat = batch['POI_catid'].to(self.device)
        input_seq_time = batch['norm_time'].to(self.device)
        user_emb = self.user_emb(input_user_id).unsqueeze(dim=1)
        user_emb_expand = user_emb.expand(-1, input_seqs.shape[1], -1)
        poi_emb = all_poi_embeddings[input_seqs]
        cat_emb = self.cat_emb(input_seq_cat)
        time_emb = self.time_embed_model(input_seq_time.to(torch.float32)).reshape(input_seqs.shape[0], input_seqs.shape[1], -1)

        fuse1 = self.fuse_embed1(user_emb_expand, poi_emb)
        fuse2 = self.fuse_embed2(cat_emb, time_emb)
        src = torch.cat([fuse1, fuse2], dim=-1)
        masks = batch['mask'].to(self.device)
        indices = (
            torch.arange(input_seqs.shape[1])
            .unsqueeze(0)
            .expand(input_seqs.shape[0], input_seqs.shape[1])
            .to(self.device)
        )
        mask = (indices >= masks.unsqueeze(1))
        src_mask = self.generate_square_subsequent_mask(batch['POI_id'].shape[1]).to(self.device)
        
        # transformer model
        src = src * math.sqrt(self.embed_size) # [batch_size, seq_len, embed_dim]
        src = self.pos_encoder(src) # [batch_size, seq_len, embed_dim]

        x = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=mask) # [batch_size, seq_len, embed_dim]
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat
    
    def forward(self, batch: Dict[str, Any], X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        
        out_poi, out_time, out_cat = self.predict(batch, X, A)

        # Compute attention map once for the entire batch
        y_pred_poi_adjusted = torch.zeros_like(out_poi)
        attn_map = self.node_attn_model(X, A)
        batch_input_seqs = batch['POI_id']
        for i in range(batch_input_seqs.shape[0]):
            traj_i_input = batch_input_seqs[i]  # input check-in pois
            for j in range(traj_i_input.shape[0]):
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + out_poi[i, j, :]
        
        return out_poi, out_time, out_cat
    

