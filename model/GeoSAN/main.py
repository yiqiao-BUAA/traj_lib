# -*- coding: utf-8 -*-
# traj_lib/model/GeoSAN/main.py
#
# GeoSAN re-implementation - CORRECTED VERSION
#
# This version fixes the critical performance issue by ensuring geographical
# information is used consistently during both training and inference.
#
# Key Corrections:
# 1.  POI Geo-data Collection: During vocabulary building, we now also create a
#     map from each POI ID to its canonical latitude and longitude.
# 2.  Geo-Embedding Pre-computation: A new non-trainable table, `geo_embedding_table`,
#     is created. Upon initialization, the model uses the GeographyEncoder to
#     compute and store the geographical embedding for every POI in the dataset.
# 3.  Unified Item Representation: An item's full representation is now always
#     the sum of its ID-based embedding and its pre-computed geographical embedding.
# 4.  Consistent Scoring: Both the `forward` (training) and `predict` (inference)
#     methods now use this unified, geography-aware item representation for scoring,
#     ensuring the model leverages what it has learned.

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple, Callable

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logger import get_logger
from utils.exargs import ConfigResolver

# ----------------------------- Load YAML ------------------------------------
model_args = ConfigResolver("./model/GeoSAN/GeoSAN.yaml").parse()
log = get_logger(__name__)

pre_views: list[str] = []
post_views: list[str] = []

# ---------------------------- Global states ---------------------------------
model: Optional[nn.Module] = None
_device: Optional[torch.device] = None


# ============================== Utilities ===================================
def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move all tensors in (possibly nested) dict to the given device."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _get_targets(batch: Dict[str, Any]) -> torch.Tensor:
    """Extract ground-truth next-POI ID: [B] long."""
    y = batch["y_POI_id"]["POI_id"]  # [B]
    return y.long()


def _make_key_padding_mask(lengths: torch.Tensor, S: int) -> torch.Tensor:
    """
    Build key padding mask for attention. True for PAD positions to be masked.
    Args:
        lengths: [B] valid lengths
        S      : sequence length
    Returns:
        mask: [B, S] bool, True for PAD
    """
    device = lengths.device
    ar = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
    valid = ar < lengths.unsqueeze(1)                # [B, S]
    return ~valid


def _infer_vocab_sizes_and_geo_map(
    train_dl: DataLoader,
    val_dl: Optional[DataLoader] = None,
    test_dl: Optional[DataLoader] = None,
) -> Tuple[int, int, int, int, Dict[int, Tuple[float, float]]]:
    """
    Infer vocab sizes, max sequence length, and a map of POI_ID -> (lat, lon) from provided splits.
    """
    num_poi: int = 1
    num_user: int = 1
    num_cat: int = 1
    max_seq: int = 1
    poi_geo_map: Dict[int, Tuple[float, float]] = {}

    log.info("Scanning dataset to build vocabulary and POI geo-map...")
    for loader in (train_dl, val_dl, test_dl):
        if loader is None:
            continue
        # Access the underlying dataset from the dataloader
        subset_dataset = loader.dataset
        # Get all samples from the subset
        samples = [subset_dataset.dataset.samples[i] for i in subset_dataset.indices]
        
        for sample in samples:
            # For sequence data
            poi_ids = sample['POI_id']
            latitudes = sample['latitude']
            longitudes = sample['longitude']
            
            num_poi = max(num_poi, int(np.max(poi_ids)))
            num_user = max(num_user, int(sample['user_id']))
            if "POI_catid" in sample and isinstance(sample["POI_catid"], np.ndarray):
                num_cat = max(num_cat, int(np.max(sample['POI_catid'])))
            max_seq = max(max_seq, len(poi_ids))

            for i in range(len(poi_ids)):
                pid = int(poi_ids[i])
                if pid != 0 and pid not in poi_geo_map:
                    poi_geo_map[pid] = (float(latitudes[i]), float(longitudes[i]))

            # For target data
            target_poi_id = int(sample['y_POI_id']['POI_id'])
            if target_poi_id != 0 and target_poi_id not in poi_geo_map:
                poi_geo_map[target_poi_id] = (float(sample['y_POI_id']['latitude']), float(sample['y_POI_id']['longitude']))
            num_poi = max(num_poi, target_poi_id)


    # Add 1 for padding/unknown index 0
    return num_poi + 1, num_user + 1, num_cat + 1, max_seq, poi_geo_map

# ======================== GeoSAN Core Components ============================
# --- Geography Encoder: From GPS to Embedding ---

def gps_to_quadkey(lat: float, lon: float, level: int) -> str:
    """
    Converts GPS coordinates to a quadkey string at a specific detail level.
    Implementation based on standard Web Mercator projection formulas.
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** level
    tile_x = int((lon + 180.0) / 360.0 * n)
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    
    quadkey = []
    for i in range(level, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tile_x & mask) != 0:
            digit += 1
        if (tile_y & mask) != 0:
            digit += 2
        quadkey.append(str(digit))
        
    return "".join(quadkey)

def quadkey_to_ngrams(quadkey: str, n: int) -> list[str]:
    """Converts a quadkey string to a list of its n-grams."""
    return [quadkey[i:i+n] for i in range(len(quadkey) - n + 1)]

class GeographyEncoder(nn.Module):
    """
    Encodes GPS coordinates into a dense vector representation as described in the GeoSAN paper.
    It first converts GPS to a quadkey, then processes its n-grams with a self-attention network.
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float, ngram_size: int, quadkey_level: int):
        super().__init__()
        self.ngram_size = ngram_size
        self.quadkey_level = quadkey_level
        self.d_model = d_model
        
        self.vocab_size = 4 ** ngram_size
        self.ngram_embedding = nn.Embedding(self.vocab_size + 1, d_model, padding_idx=0)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.base4_powers = torch.tensor([4**i for i in range(ngram_size-1, -1, -1)], dtype=torch.long)
    
    def _ngrams_to_indices(self, ngrams: list[list[str]], device: torch.device) -> torch.Tensor:
        """Converts lists of n-gram strings to a padded tensor of indices."""
        max_len = max(len(sublist) for sublist in ngrams) if ngrams else 0
        if max_len == 0:
            return torch.zeros(len(ngrams), 0, dtype=torch.long, device=device)
        
        indices = torch.zeros(len(ngrams), max_len, dtype=torch.long, device=device)
        self.base4_powers = self.base4_powers.to(device)

        for i, ngram_list in enumerate(ngrams):
            if not ngram_list: continue
            digits = torch.tensor([[int(c) for c in ng] for ng in ngram_list], dtype=torch.long, device=device)
            ngram_indices = (digits.float() @ self.base4_powers.float()).long() + 1
            indices[i, :len(ngram_indices)] = ngram_indices
            
        return indices

    def forward(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lat (torch.Tensor): Latitudes, shape [B, S] or [N].
            lon (torch.Tensor): Longitudes, shape [B, S] or [N].
        Returns:
            torch.Tensor: Geographical embeddings, shape [B, S, D] or [N, D].
        """
        is_batched_seq = lat.dim() == 2
        if is_batched_seq:
            B, S = lat.shape
        else:
            B, S = lat.shape[0], 1
        
        device = lat.device
        flat_lat, flat_lon = lat.reshape(-1).cpu().numpy(), lon.reshape(-1).cpu().numpy()
        
        all_ngrams = [quadkey_to_ngrams(gps_to_quadkey(la, lo, self.quadkey_level), self.ngram_size) for la, lo in zip(flat_lat, flat_lon)]
        
        ngram_indices = self._ngrams_to_indices(all_ngrams, device)
        embedded_ngrams = self.ngram_embedding(ngram_indices)
        padding_mask = (ngram_indices == 0)

        encoded_output = self.transformer_encoder(embedded_ngrams, src_key_padding_mask=padding_mask)
        
        encoded_output.masked_fill_(padding_mask.unsqueeze(-1), 0.0)
        summed = encoded_output.sum(dim=1)
        non_pad_count = (~padding_mask).sum(dim=1).unsqueeze(1).clamp(min=1)
        pooled_output = summed / non_pad_count
        
        final_shape = (B, S, self.d_model) if is_batched_seq else (B, self.d_model)
        geo_embeddings = self.layer_norm(pooled_output.view(final_shape))
        
        return geo_embeddings

class TargetAwareAttentionDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, target_emb: torch.Tensor, seq_output: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        is_single_target = target_emb.dim() == 2
        if is_single_target:
            target_emb = target_emb.unsqueeze(1)

        attn_output, _ = self.attention(query=target_emb, key=seq_output, value=seq_output, key_padding_mask=key_padding_mask)
        
        output = self.layer_norm(attn_output + target_emb)
        
        if is_single_target:
            output = output.squeeze(1)
        return output

class GeoSAN(nn.Module):
    def __init__(self, num_poi, num_user, max_seq_len, poi_geo_map: Dict[int, Tuple[float, float]], **cfg):
        super().__init__()
        self.num_poi = num_poi
        d_model = cfg['d_model']
        n_heads = cfg['n_heads']
        n_layers = cfg['n_layers']
        dropout = cfg['dropout']

        self.use_user_emb = cfg['use_user_emb']
        self.use_time_emb = cfg['use_time_emb']
        self.use_target_aware_decoder = cfg['use_target_aware_decoder']
        
        self.poi_embedding = nn.Embedding(num_poi, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        if self.use_user_emb:
            self.user_embedding = nn.Embedding(num_user, d_model, padding_idx=0)
        if self.use_time_emb:
            self.time_embedding = nn.Embedding(168 + 1, d_model, padding_idx=0)

        self.geography_encoder = GeographyEncoder(
            d_model=d_model, n_heads=n_heads,
            n_layers=cfg['geo_encoder_n_layers'], dropout=dropout,
            ngram_size=cfg['ngram_size'], quadkey_level=cfg['quadkey_level']
        )
        
        input_dim = d_model * 2 # POI + Pos
        if self.use_user_emb: input_dim += d_model
        if self.use_time_emb: input_dim += d_model
        # Note: Geo emb is now added after fusion
        self.fusion_layer = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if self.use_target_aware_decoder:
            self.decoder = TargetAwareAttentionDecoder(d_model, n_heads)

        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)
        
        # --- Pre-compute Geo Embeddings for all POIs ---
        # Create a non-trainable buffer to store pre-computed geo embeddings
        self.register_buffer('geo_embedding_table', torch.zeros(num_poi, d_model))
        self._precompute_geo_embeddings(poi_geo_map, d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'geo_embedding_table' not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @torch.no_grad()
    def _precompute_geo_embeddings(self, poi_geo_map: Dict[int, Tuple[float, float]], d_model: int):
        """
        Uses the GeographyEncoder to compute geo embeddings for all POIs and store them.
        """
        log.info(f"Pre-computing geographical embeddings for {len(poi_geo_map)} POIs...")
        device = next(self.geography_encoder.parameters()).device
        
        # Prepare batch of coordinates
        poi_ids = sorted(poi_geo_map.keys())
        lats = torch.tensor([poi_geo_map[pid][0] for pid in poi_ids], dtype=torch.float32, device=device)
        lons = torch.tensor([poi_geo_map[pid][1] for pid in poi_ids], dtype=torch.float32, device=device)

        # Encode in batches to avoid OOM
        batch_size = 512
        for i in tqdm(range(0, len(poi_ids), batch_size), desc="Encoding POIs"):
            batch_ids = poi_ids[i:i+batch_size]
            batch_lats = lats[i:i+batch_size]
            batch_lons = lons[i:i+batch_size]
            
            # [batch_size, D]
            geo_embs = self.geography_encoder(batch_lats, batch_lons)
            self.geo_embedding_table[torch.tensor(batch_ids, dtype=torch.long)] = geo_embs.cpu()
        
        # Move final table to the correct device
        self.geo_embedding_table = self.geo_embedding_table.to(device)
        log.info("Finished pre-computing geographical embeddings.")
    
    def get_full_item_representation(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Gets the full representation of an item by combining its ID and geo embeddings.
        """
        # [B, K, D]
        id_emb = self.poi_embedding(item_ids)
        # [B, K, D]
        geo_emb = F.embedding(item_ids, self.geo_embedding_table)
        return id_emb + geo_emb

    def encode_sequence(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input sequence into a series of hidden states."""
        poi_ids = batch["POI_id"].long()
        lengths = batch["mask"].long()
        B, S = poi_ids.shape

        # --- Get ID-based and Positional Embeddings ---
        # [B, S, D]
        poi_emb = self.poi_embedding(poi_ids)
        pos_ids = torch.arange(S, device=poi_ids.device).unsqueeze(0).expand(B, S)
        # [B, S, D]
        pos_emb = self.pos_embedding(pos_ids)

        embeddings_to_concat = [poi_emb, pos_emb]
        if self.use_user_emb:
            user_ids = batch["user_id"].long()
            user_emb = self.user_embedding(user_ids).unsqueeze(1).expand(B, S, -1)
            embeddings_to_concat.append(user_emb)
        if self.use_time_emb:
            hour_of_week = (batch["timestamps"].long() // 3600) % 168
            time_emb = self.time_embedding(hour_of_week)
            embeddings_to_concat.append(time_emb)

        # [B, S, D_in] -> [B, S, D]
        fused_emb = self.fusion_layer(torch.cat(embeddings_to_concat, dim=-1))

        # --- Add Geographical Embeddings ---
        # [B, S, D]
        geo_emb = F.embedding(poi_ids, self.geo_embedding_table)
        final_emb = fused_emb + geo_emb
        final_emb = self.dropout(final_emb)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=final_emb.device)
        padding_mask = _make_key_padding_mask(lengths, S)
        
        # [B, S, D] -> [B, S, D]
        seq_output = self.encoder(final_emb, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        return seq_output, padding_mask

    def forward(self, batch: Dict[str, Any], pos_targets: torch.Tensor, neg_targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for TRAINING.
        """
        # [B, S, D], [B, S]
        seq_output, padding_mask = self.encode_sequence(batch)
        B = seq_output.size(0)
        
        # [B]
        last_indices = torch.clamp(batch["mask"].long() - 1, min=0)
        # [B, 1, D] -> [B, D]
        last_hidden_state = seq_output.gather(dim=1, index=last_indices.view(B, 1, 1).expand(-1, -1, seq_output.size(-1))).squeeze(1)
        
        # --- Get FULL Target Embeddings (ID + Geo) ---
        # [B, D]
        pos_target_emb = self.get_full_item_representation(pos_targets)
        # [B, K, D]
        neg_target_emb = self.get_full_item_representation(neg_targets)
        
        # --- Decode and Score ---
        if self.use_target_aware_decoder:
            # [B, D]
            pos_context = self.decoder(pos_target_emb, seq_output, padding_mask)
            
            _, K, D = neg_target_emb.shape
            neg_target_emb_flat = neg_target_emb.view(B*K, D)
            seq_output_rep = seq_output.unsqueeze(1).expand(-1, K, -1, -1).reshape(B*K, seq_output.size(1), D)
            padding_mask_rep = padding_mask.unsqueeze(1).expand(-1, K, -1).reshape(B*K, padding_mask.size(1))
            neg_context_flat = self.decoder(neg_target_emb_flat, seq_output_rep, padding_mask_rep)
            neg_context = neg_context_flat.view(B, K, D)

            query_vec = self.output_norm(last_hidden_state)
            pos_scores = (query_vec.unsqueeze(1) @ pos_context.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            neg_scores = (query_vec.unsqueeze(1) @ neg_context.transpose(1, 2)).squeeze(1)
        else:
            query_vec = self.output_norm(last_hidden_state)
            pos_scores = (query_vec * pos_target_emb).sum(dim=-1)
            neg_scores = (query_vec.unsqueeze(1) * neg_target_emb).sum(dim=-1)

        return pos_scores, neg_scores

    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass for INFERENCE.
        """
        # [B, S, D], [B, S]
        seq_output, padding_mask = self.encode_sequence(batch)
        B = seq_output.size(0)

        # [B] -> [B, D]
        last_indices = torch.clamp(batch["mask"].long() - 1, min=0)
        last_hidden_state = seq_output.gather(dim=1, index=last_indices.view(B, 1, 1).expand(-1, -1, seq_output.size(-1))).squeeze(1)
        last_hidden_state = self.output_norm(last_hidden_state)
        
        # --- Scoring all POIs using the FULL item representation ---
        # [V, D]
        full_item_embeddings = self.get_full_item_representation(torch.arange(self.num_poi, device=last_hidden_state.device))
        
        # [B, D] @ [D, V] -> [B, V]
        logits = last_hidden_state @ full_item_embeddings.t()
        
        return logits


# ============================ Train / Inference =============================
def _build_model(
    train_dl: DataLoader,
    val_dl: Optional[DataLoader],
    test_dl: Optional[DataLoader],
    **cfg: Any,
) -> nn.Module:
    """Construct model with vocab sizes inferred from dataloader."""
    global model, _device

    num_poi, num_user, num_cat, max_seq, poi_geo_map = _infer_vocab_sizes_and_geo_map(train_dl, val_dl, test_dl)
    log.info("[GeoSAN] Vocab: V=%d, U=%d, C=%d, S_max=%d", num_poi, num_user, num_cat, max_seq)
    log.info(f"[GeoSAN] Found {len(poi_geo_map)} unique POIs with geo-coordinates.")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    built = GeoSAN(
        num_poi=num_poi, num_user=num_user, max_seq_len=max_seq, poi_geo_map=poi_geo_map,
        **model_args
    ).to(_device)

    model = built
    return built

# The public API functions remain unchanged in their signature
def init(dataloader: Any, **cfg: Any) -> None:
    _build_model(dataloader, None, None, **cfg)

def train_one_epoch(dataloader: Any, **cfg: Any) -> None:
    assert model is not None, "Call init(dataloader) before train_one_epoch()."
    for _ in _train_impl(dataloader, None, {}, one_epoch=True, eval_funcs=None, **cfg):
        pass

def train(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    view_value: Dict[str, Any],
    eval_funcs: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]] = None,
    **cfg: Any,
) -> Iterable[Dict[str, float]]:
    global model
    if model is None:
        _build_model(train_dataloader, val_dataloader, None, **cfg)

    return _train_impl(train_dataloader, val_dataloader, view_value, one_epoch=False, eval_funcs=eval_funcs, **cfg)


def _train_impl(
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    view_value: Dict[str, Any],
    one_epoch: bool,
    eval_funcs: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]],
    **cfg: Any,
) -> Iterable[Dict[str, Any]]:
    """
    Main training loop implementing negative sampling and importance sampling loss.
    """
    global model, _device
    assert model is not None and isinstance(model, GeoSAN)
    device = _device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=model_args["lr"], weight_decay=model_args["weight_decay"])
    grad_clip: float = float(model_args["grad_clip"])
    run_epochs: int = 1 if one_epoch else int(model_args["epochs"])
    num_neg = int(model_args["num_negative_samples"])
    temperature = float(model_args["loss_temperature"])
    
    for ep in range(run_epochs):
        model.train()
        total_loss: float = 0.0
        total_cnt: int = 0

        for batch in train_dataloader:
            batch = _batch_to_device(batch, device)
            targets = _get_targets(batch)
            bs = targets.size(0)

            neg_samples = torch.randint(1, model.num_poi, (bs, num_neg), device=device)
            pos_scores, neg_scores = model(batch, targets, neg_samples)
            
            loss = -F.logsigmoid(pos_scores).mean()
            if num_neg > 0:
                with torch.no_grad():
                    weights = F.softmax(neg_scores / temperature, dim=-1)
                neg_loss = - (weights * F.logsigmoid(-neg_scores)).sum() / bs
                loss += neg_loss
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            total_loss += float(loss.item()) * bs
            total_cnt += bs

        avg_loss = total_loss / max(1, total_cnt)
        log.info("[GeoSAN][train] epoch %d/%d  loss=%.6f", ep + 1, run_epochs, avg_loss)

        if eval_funcs and val_dataloader is not None:
            inference_res = _inference_on_split(val_dataloader, view_value)
            preds = inference_res['pred']
            gts = inference_res['gts']
            scores: Dict[str, float] = {name: float(fn(preds, gts)) for name, fn in eval_funcs.items()}
            
            # metrics_dict = {"loss": avg_loss}
            metrics_dict: Dict[str, float] = {}
            metrics_dict.update(scores)
            yield [metrics_dict]


@torch.no_grad()
def _inference_on_split(
    split_dataloader: DataLoader,
    view_value: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference on a split; outputs on CPU for framework metrics."""
    global model, _device
    assert model is not None and isinstance(model, GeoSAN)
    device = _device or torch.device("cuda"if torch.cuda.is_available() else "cpu")
    model.eval()

    probs_list: list[torch.Tensor] = []
    gts_list: list[torch.Tensor] = []

    for batch in split_dataloader:
        batch = _batch_to_device(batch, device)
        targets = _get_targets(batch)
        logits = model.predict(batch)
        probs = torch.softmax(logits, dim=-1)
        probs_list.append(probs.cpu())
        gts_list.append(targets.cpu())

    preds = torch.cat(probs_list, dim=0)  # [N, V]
    gts = torch.cat(gts_list, dim=0)      # [N]
    return {'pred': preds.numpy(), 'gts': gts.numpy()}


@torch.no_grad()
def inference(
    test_dataloader: DataLoader,
    view_value: Dict[str, Any],
    **cfg: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Framework-facing inference API."""
    global model
    if model is None:
        log.warning("Model not initialized. Building model from test_dataloader. Vocab size may be incomplete.")
        _build_model(test_dataloader, None, test_dataloader, **cfg)
    
    return _inference_on_split(test_dataloader, view_value)