from typing import Optional, Tuple, Callable, Union, cast, Any

import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils.exargs import ParseDict
from .ROTAN_utils import rotate, rotate_batch


class GPSEncoder(nn.Module):
    def __init__(
        self, embed_size: int, nhead: int, nhid: int, nlayers: int, dropout: float
    ):
        super(GPSEncoder, self).__init__()
        encoder_layers = TransformerEncoderLayer(
            embed_size, nhead, nhid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.norm = nn.LayerNorm(embed_size)

    # s*l*d
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src = src * math.sqrt(self.embed_size)
        x = self.transformer_encoder(src, mask=mask, src_key_padding_mask=src_key_mask)
        x = torch.mean(x, -2)
        return self.norm(x)


def t2v(
    tau: torch.Tensor,
    f: Callable[..., torch.Tensor],
    w: torch.Tensor,
    b: torch.Tensor,
    w0: torch.Tensor,
    b0: torch.Tensor,
    arg: Optional[torch.Tensor] = None,
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
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)


class RightPositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model: int, dropout: float, max_len: int = 600):
        super(RightPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = cast(torch.Tensor, self.pe)[:, : x.size(1)]
        x = x + pe.requires_grad_(False)
        return self.dropout(x)


class CosineActivation(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)


class OriginTime2Vec(nn.Module):
    def __init__(self, activation: str, out_dim: int):
        super(OriginTime2Vec, self).__init__()
        self.l1: Union[SineActivation, CosineActivation]
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x: torch.Tensor):
        fea = x.view(-1, 1)
        return self.l1(fea)


class ROTAN(nn.Module):

    def __init__(self, args: ParseDict):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.device = args.device

        self.user_embed_dim = args.user_embed_dim
        self.poi_embed_dim = args.poi_embed_dim
        self.time_embed_dim = args.time_embed_dim
        self.gps_embed_dim = args.gps_embed_dim

        self.gps_embed_model = nn.Embedding(
            num_embeddings=4096, embedding_dim=self.gps_embed_dim
        )
        self.user_embed_model = nn.Embedding(
            num_embeddings=args.num_users, embedding_dim=self.user_embed_dim
        )
        self.poi_embed_model = nn.Embedding(
            num_embeddings=args.num_pois, embedding_dim=self.poi_embed_dim
        )

        self.time_embed_model_user = OriginTime2Vec(
            "sin", int(0.5 * (self.user_embed_dim + self.poi_embed_dim))
        )
        self.time_embed_model_user_tgt = OriginTime2Vec(
            "sin", int(0.5 * (self.user_embed_dim + self.poi_embed_dim))
        )
        # time_embed_model_user = TimeEmbeddings(96,int(0.5*(self.user_embed_dim+self.poi_embed_dim)))
        self.time_embed_model_user_day = OriginTime2Vec(
            "sin", int(0.5 * (self.user_embed_dim + self.poi_embed_dim))
        )
        self.time_embed_model_user_day_tgt = OriginTime2Vec(
            "sin", int(0.5 * (self.user_embed_dim + self.poi_embed_dim))
        )

        self.time_embed_model_poi = OriginTime2Vec("sin", 2 * self.time_embed_dim)
        self.time_embed_model_poi_tgt = OriginTime2Vec("sin", 2 * self.time_embed_dim)
        # time_embed_model_poi = TimeEmbeddings(96,self.time_embed_dim)
        self.time_embed_model_poi_day = OriginTime2Vec("sin", 2 * self.time_embed_dim)
        self.time_embed_model_poi_day_tgt = OriginTime2Vec(
            "sin", 2 * self.time_embed_dim
        )

        self.gps_encoder = GPSEncoder(
            self.gps_embed_dim, 1, 2 * self.gps_embed_dim, 2, 0.3
        )

        self.n_head = args.transformer_nhead
        # 0.4
        self.dropout = args.transformer_dropout
        self.n_layers = args.transformer_nlayers
        # 1024
        self.hidden_size = args.transformer_nhid

        self.pos_encoder1 = RightPositionalEncoding(
            self.user_embed_dim + self.poi_embed_dim, self.dropout
        )
        self.pos_encoder2 = RightPositionalEncoding(
            self.poi_embed_dim + self.gps_embed_dim, self.dropout
        )
        # self.pos_encoder = RightPositionalEncoding(args.user_embed_dim+args.poi_embed_dim+args.gps_embed_dim+user_time_dim,dropout)

        encoder_layers1 = TransformerEncoderLayer(
            self.user_embed_dim + self.poi_embed_dim,
            self.n_head,
            self.hidden_size,
            self.dropout,
            batch_first=True,
        )
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, self.n_layers)

        encoder_layers2 = TransformerEncoderLayer(
            self.poi_embed_dim + self.gps_embed_dim,
            self.n_head,
            self.hidden_size,
            self.dropout,
            batch_first=True,
        )
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, self.n_layers)

        self.decoder_poi1 = nn.Linear(
            self.user_embed_dim + 2 * self.poi_embed_dim, args.num_pois
        )
        self.decoder_poi2 = nn.Linear(
            self.poi_embed_dim + 2 * self.gps_embed_dim, args.num_pois
        )

        self.criterion_poi = nn.CrossEntropyLoss(ignore_index=0)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi1.bias.data.zero_()
        self.decoder_poi1.weight.data.uniform_(-initrange, initrange)
        self.decoder_poi2.bias.data.zero_()
        self.decoder_poi2.weight.data.uniform_(-initrange, initrange)

    def compute_poi_prob(
        self,
        src1: torch.Tensor,
        src2: torch.Tensor,
        src_mask: torch.Tensor,
        target_hour: torch.Tensor,
        target_day: torch.Tensor,
        poi_embeds: torch.Tensor,
        gps_embeds: torch.Tensor,
        src_key_mask: torch.Tensor,
    ) -> torch.Tensor:

        src1 = src1 * math.sqrt(self.user_embed_dim + self.poi_embed_dim)
        src1 = self.pos_encoder1(src1)
        src1 = self.transformer_encoder1(
            src=src1, mask=src_mask, src_key_padding_mask=src_key_mask
        )

        user_time_dim = int(0.5 * (self.user_embed_dim + self.poi_embed_dim))

        src1_hour = rotate_batch(
            src1, target_hour[:, :, :user_time_dim], user_time_dim, self.device
        )
        src1_day = rotate_batch(
            src1, target_day[:, :, :user_time_dim], user_time_dim, self.device
        )

        """ src1_hour = torch.cat((src1,target_hour[:,:,:user_time_dim]),dim=-1)
        src1_day = torch.cat((src1,target_day[:,:,:user_time_dim]),dim=-1) """

        src1 = 0.7 * src1_hour + 0.3 * src1_day
        src1 = torch.cat((src1, poi_embeds), dim=-1)

        out_poi_prob1 = self.decoder_poi1(src1)

        src2 = src2 * math.sqrt(self.poi_embed_dim + self.gps_embed_dim)
        src2 = self.pos_encoder2(src2)
        src2 = self.transformer_encoder2(
            src=src2, mask=src_mask, src_key_padding_mask=src_key_mask
        )

        src2_hour = rotate_batch(
            src2,
            target_hour[:, :, user_time_dim:],
            2 * self.time_embed_dim,
            self.device,
        )
        src2_day = rotate_batch(
            src2, target_day[:, :, user_time_dim:], 2 * self.time_embed_dim, self.device
        )
        """ src2_hour = torch.cat((src2,target_hour[:,:,user_time_dim:]),dim=-1)
        src2_day = torch.cat((src2,target_day[:,:,user_time_dim:]),dim=-1) """

        src2 = 0.7 * src2_hour + 0.3 * src2_day
        src2 = torch.cat((src2, gps_embeds), dim=-1)

        out_poi_prob2 = self.decoder_poi2(src2)

        out_poi_prob = 0.7 * out_poi_prob1 + 0.3 * out_poi_prob2

        return out_poi_prob

    def handle_sequence(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        poi_id, norm_time, day_time = (
            batch_data["POI_id"],
            batch_data["norm_time"],
            batch_data["day_time"],
        )
        user_id = (
            batch_data["user_id"]
            .unsqueeze(dim=1)
            .expand(poi_id.shape[0], poi_id.shape[1])
        )

        seq_len = batch_data["mask"]

        y_poi_id, y_norm_time, y_day_time = (
            batch_data["y_POI_id"]["POI_id"],
            batch_data["y_POI_id"]["norm_time"],
            batch_data["y_POI_id"]["day_time"],
        )

        B, L = batch_data["POI_id"].size()
        y_poi_seq = torch.full((B, L), 0, dtype=torch.long, device=self.device)
        y_norm_time_seq = torch.full((B, L), 0, dtype=torch.float, device=self.device)
        y_day_time_seq = torch.full((B, L), 0, dtype=torch.float, device=self.device)

        for i in range(B):
            end = seq_len[i].item()
            y_poi_seq[i, :end] = torch.cat(
                (poi_id[i, 1:end], y_poi_id[i].unsqueeze(dim=-1)), dim=-1
            )
            y_norm_time_seq[i, :end] = torch.cat(
                (norm_time[i, 1:end], y_norm_time[i].unsqueeze(dim=-1)), dim=-1
            )
            y_day_time_seq[i, :end] = torch.cat(
                (day_time[i, 1:end], y_day_time[i].unsqueeze(dim=-1)), dim=-1
            )

        batch_data["user_id"] = user_id
        batch_data["y_POI_id"]["POI_id"] = y_poi_seq
        batch_data["y_POI_id"]["norm_time"] = y_norm_time_seq
        batch_data["y_POI_id"]["day_time"] = y_day_time_seq

        mask = batch_data["mask"]
        indices = (
            torch.arange(user_id.shape[1])
            .unsqueeze(0)
            .expand(user_id.shape[0], user_id.shape[1])
            .to(self.device)
        )
        mask = (indices >= mask.unsqueeze(1)).to(torch.bool)
        batch_data["mask"] = mask

        return batch_data

    def get_predict(
        self, batch_data: dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_data = self.handle_sequence(batch_data)

        seq_len = batch_data["POI_id"].size(1)
        mask = (torch.triu(torch.ones(seq_len, seq_len))).transpose(0, 1)
        src_mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .to(self.device)
        )

        (
            x1,
            x2,
            batch_target_time,
            batch_target_day,
            poi_embeds_padded,
            gps_embeds_padded,
        ) = self.get_rotation_and_loss(batch_data)
        y_poi = batch_data["y_POI_id"]["POI_id"]

        y_pred_poi = self.compute_poi_prob(
            x1,
            x2,
            src_mask,
            batch_target_time,
            batch_target_day,
            poi_embeds_padded,
            gps_embeds_padded,
            batch_data["mask"],
        )

        return y_pred_poi, y_poi

    def forward(self, batch_data: dict[str, Any]) -> torch.Tensor:
        y_pred_poi, y_poi = self.get_predict(batch_data)
        loss_poi = self.criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
        return loss_poi

    def predict(self, batch_data: dict[str, Any]) -> torch.Tensor:
        y_pred_poi, _ = self.get_predict(batch_data)
        # original mask: 0, 0, 0, 1, 1, ……, 1
        batch_data["mask"] = y_pred_poi.shape[1] - torch.sum(batch_data["mask"], dim=-1)
        batch_indices = torch.arange(batch_data["mask"].shape[0])
        y_pred = y_pred_poi[batch_indices, batch_data["mask"] - 1]
        return y_pred

    def get_rotation_and_loss(
        self, batch_data: dict[str, Any]
    ) -> Tuple[torch.Tensor, ...]:
        # Parse sample
        u_id, poi_id, time, day_time, gps_id = (
            batch_data["user_id"],
            batch_data["POI_id"],
            batch_data["norm_time"].to(torch.float),
            batch_data["day_time"].to(torch.float),
            batch_data["quad_key"],
        )
        target_time, target_day_time = batch_data["y_POI_id"]["norm_time"].to(
            torch.float
        ), batch_data["y_POI_id"]["day_time"].to(torch.float)

        gps_embeddings = self.gps_embed_model(gps_id)
        # batch_size, seq_len, LEN_QUADKEY, gps_embed_dim
        gps_embeddings = self.gps_encoder(
            gps_embeddings.reshape(-1, gps_embeddings.size(2), self.gps_embed_dim)
        )
        gps_embeddings = gps_embeddings.reshape(
            poi_id.size(0), poi_id.size(1), self.gps_embed_dim
        )

        # User to embedding
        user_embeddings = self.user_embed_model(u_id)
        # user_embedding = torch.squeeze(user_embedding)
        # user_embeddings = user_embedding.repeat(len(u_id),1).to(self.device)

        user_times = self.time_embed_model_user(time).reshape(
            poi_id.shape[0], poi_id.shape[1], -1
        )
        user_day_times = self.time_embed_model_user_day(day_time).reshape(
            poi_id.shape[0], poi_id.shape[1], -1
        )

        seq_poi_embeddings = self.poi_embed_model(poi_id)
        # seq_poi_embeddings = torch.index_select(poi_pre_embeddings,0,input_seq)
        poi_embeds = seq_poi_embeddings

        user_next_times = self.time_embed_model_user_tgt(target_time).reshape(
            poi_id.shape[0], poi_id.shape[1], -1
        )
        user_next_day_times = self.time_embed_model_user_day_tgt(
            target_day_time
        ).reshape(poi_id.shape[0], poi_id.shape[1], -1)

        poi_next_times = self.time_embed_model_poi_tgt(target_time).reshape(
            poi_id.shape[0], poi_id.shape[1], -1
        )
        poi_next_day_times = self.time_embed_model_poi_day_tgt(target_day_time).reshape(
            poi_id.shape[0], poi_id.shape[1], -1
        )

        poi_times = self.time_embed_model_poi(time).reshape(
            poi_id.shape[0], poi_id.shape[1], -1
        )
        poi_day_times = self.time_embed_model_poi_day(day_time).reshape(
            poi_id.shape[0], poi_id.shape[1], -1
        )

        user_embeddings = torch.cat((user_embeddings, seq_poi_embeddings), dim=-1)

        user_rotate_hour = rotate(
            user_embeddings,
            user_times,
            int(0.5 * (self.user_embed_dim + self.poi_embed_dim)),
            self.device,
        )
        user_rotate_day = rotate(
            user_embeddings,
            user_day_times,
            int(0.5 * (self.user_embed_dim + self.poi_embed_dim)),
            self.device,
        )

        user_rotate = 0.7 * user_rotate_hour + 0.3 * user_rotate_day

        seq_poi_embeddings = torch.cat((seq_poi_embeddings, gps_embeddings), dim=-1)
        poi_rotate_hour = rotate(
            seq_poi_embeddings, poi_times, 2 * self.time_embed_dim, self.device
        )
        poi_rotate_day = rotate(
            seq_poi_embeddings, poi_day_times, 2 * self.time_embed_dim, self.device
        )

        poi_rotate = 0.7 * poi_rotate_hour + 0.3 * poi_rotate_day

        seq_embedding1 = user_rotate
        # seq_embedding1 = torch.cat((user_embeddings,seq_poi_embeddings,input_seq_gps_embeddings,user_times),dim=-1)
        seq_embedding2 = poi_rotate
        seq_embedding3 = torch.cat((user_next_times, poi_next_times), dim=-1)
        seq_embedding4 = torch.cat((user_next_day_times, poi_next_day_times), dim=-1)

        return (
            seq_embedding1,
            seq_embedding2,
            seq_embedding3,
            seq_embedding4,
            poi_embeds,
            gps_embeddings,
        )
