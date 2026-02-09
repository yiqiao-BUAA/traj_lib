import math
from typing import Callable, Iterable, Tuple, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.exargs import ConfigResolver, ParseDict
from utils.logger import get_logger
from utils.GPU_find import find_gpu

logger = get_logger(__name__)

model_args = ConfigResolver("./model/STRNN/STRNN.yaml").parse()
model, optimizer = None, None
pre_views = ["STRNN_preview"]
post_views = ["STRNN_postview"]

device = find_gpu()

class STRNN(nn.Module):
    def __init__(self, config: ParseDict, data_feature: dict[str, int]) -> None:
        super(STRNN, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.device = device
        self.loc_size = data_feature["loc_size"]
        self.uid_size = data_feature["uid_size"]
        self.lw_time = 0.0
        self.up_time = data_feature["tim_size"] - 1
        self.lw_loc = 0.0
        self.up_loc = data_feature["distance_upper"]

        # Parameters
        self.h0 = nn.Parameter(torch.randn(size=[self.hidden_size, 1]))  # h0
        self.weight_ih = nn.Parameter(
            torch.randn(size=[self.hidden_size, self.hidden_size])
        )  # C
        self.weight_th_upper = nn.Parameter(
            torch.randn(size=[self.hidden_size, self.hidden_size])
        )  # T Tu
        self.weight_th_lower = nn.Parameter(
            torch.randn(size=[self.hidden_size, self.hidden_size])
        )  # T Tl
        self.weight_sh_upper = nn.Parameter(
            torch.randn(size=[self.hidden_size, self.hidden_size])
        )  # S Su
        self.weight_sh_lower = nn.Parameter(
            torch.randn(size=[self.hidden_size, self.hidden_size])
        )  # S Sl

        # Embeddings
        self.location_weight = nn.Embedding(self.loc_size, self.hidden_size)
        self.permanet_weight = nn.Embedding(self.uid_size, self.hidden_size)

        # (kept for compatibility, but not used at the output of forward)
        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()  # init weights

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(
        self, td_upper: torch.Tensor,
        td_lower: torch.Tensor,
        ld_upper: torch.Tensor,
        ld_lower: torch.Tensor,
        current_loc: torch.Tensor,
        loc_len: list[int]
    ) -> torch.Tensor:
        """
        td_* / ld_*: Float tensor, shape [B, L_max]; values are per-step scalars used to mix upper/lower weights.
        current_loc: Long tensor, shape [B, L_max] (location ids)
        loc_len: list[int] with actual lengths per sample (<= L_max)
        Returns: hidden representation [B, H]
        """
        B = current_loc.shape[0]
        H = self.hidden_size
        device = self.device

        # Ensure tensors are on correct device and dtype
        td_upper = td_upper.to(device).float()
        td_lower = td_lower.to(device).float()
        ld_upper = ld_upper.to(device).float()
        ld_lower = ld_lower.to(device).float()
        current_loc = current_loc.to(device).long()

        output = []
        eps = 1e-9
        for i in range(B):
            L = (
                int(loc_len[i])
                if not torch.is_tensor(loc_len)
                else int(loc_len[i].item())
            )
            if L <= 0:
                # fallback: use user vector only
                usr_vec = torch.mm(self.weight_ih, self.h0)  # [H,1]
                hx = usr_vec.reshape(1, H)
                output.append(hx)
                continue

            # Scalars for each step j
            ttd = [
                (
                    self.weight_th_upper * td_upper[i, j]
                    + self.weight_th_lower * td_lower[i, j]
                )
                / (td_upper[i, j] + td_lower[i, j] + eps)
                for j in range(L)
            ]  # each [H,H]

            sld = [
                (
                    self.weight_sh_upper * ld_upper[i, j]
                    + self.weight_sh_lower * ld_lower[i, j]
                )
                / (ld_upper[i, j] + ld_lower[i, j] + eps)
                for j in range(L)
            ]  # each [H,H]

            # Locations embeddings for the first L positions of sample i
            loc_ids = current_loc[i, :L]  # [L]
            loc = self.location_weight(loc_ids).unsqueeze(2)  # [L,H,1]

            # Sum over j: sld[j] @ (ttd[j] @ loc[j])
            seq_terms = []
            for j in range(L):
                term = torch.mm(sld[j], torch.mm(ttd[j], loc[j]))  # [H,1]
                seq_terms.append(term.unsqueeze(0))  # [1,H,1]
            loc_vec = torch.sum(torch.cat(seq_terms, dim=0), dim=0)  # [H,1]

            usr_vec = torch.mm(self.weight_ih, self.h0)  # [H,1]
            hx = (loc_vec + usr_vec).reshape(1, H)  # [1,H]
            output.append(hx)

        output_tensor = torch.cat(output, dim=0)  # [B,H]
        output_tensor = torch.softmax(output_tensor, dim=1)
        return output_tensor

    def calculate_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        device = self.device

        # --- Prepare batch fields ---
        # user ids
        user = batch["user_id"]
        if not torch.is_tensor(user):
            user = torch.as_tensor(user, dtype=torch.long, device=device)
        else:
            user = user.to(device).long()

        # target ids and timestamps
        dst = batch["y_POI_id"]["POI_id"]
        if not torch.is_tensor(dst):
            dst = torch.as_tensor(dst, dtype=torch.long, device=device)
        else:
            dst = dst.to(device).long()

        dst_time = batch["y_POI_id"]["timestamps"]
        if not torch.is_tensor(dst_time):
            dst_time = torch.as_tensor(dst_time, device=device)
        else:
            dst_time = dst_time.to(device)

        # sequence fields
        ts = batch["timestamps"]
        if not torch.is_tensor(ts):
            ts = torch.as_tensor(ts, device=device)
        else:
            ts = ts.to(device)

        ld = batch["current_dis"]
        if not torch.is_tensor(ld):
            ld = torch.as_tensor(ld, device=device)
        else:
            ld = ld.to(device)

        current_loc = batch["POI_id"]
        if not torch.is_tensor(current_loc):
            current_loc = torch.as_tensor(current_loc, device=device)
        else:
            current_loc = current_loc.to(device)

        loc_len = batch["mask"]
        if torch.is_tensor(loc_len):
            loc_len_list = loc_len.tolist()
        else:
            loc_len_list = list(loc_len)

        # --- Build td/ld per step ---
        # td[i, j] = dst_time[i] - ts[i, j]
        if ts.ndim == 1:
            ts = ts.unsqueeze(1)
        td = dst_time.unsqueeze(1) - ts  # [B, L]
        td_upper = (self.up_time - td).float()
        td_lower = td.float()

        if ld.ndim == 1:
            ld = ld.unsqueeze(1)
        ld_upper = (self.up_loc - ld).float()
        ld_lower = ld.float()

        # --- Hidden representation ---
        h_tq = self.forward(
            td_upper, td_lower, ld_upper, ld_lower, current_loc, loc_len_list
        )  # [B,H]
        # print(h_tq)

        # --- Score of target (positive sample) ---
        p_u = self.permanet_weight(user)  # [B,H]
        q_v = self.location_weight(dst)  # [B,H]
        scores = h_tq + p_u  # [B,H]
        scores = (scores * q_v).sum(dim=1)  # [B]

        # Logistic loss for positive samples: mean softplus(-score)
        loss = F.softplus(-scores).mean()
        return loss

    def predict(self, batch: dict[str, Any]) -> torch.Tensor:
        device = self.device
        user = batch["user_id"]
        if not torch.is_tensor(user):
            user = torch.as_tensor(user, dtype=torch.long, device=device)
        else:
            user = user.to(device).long()

        dst_time = batch["y_POI_id"]["timestamps"]
        if not torch.is_tensor(dst_time):
            dst_time = torch.as_tensor(dst_time, device=device)
        else:
            dst_time = dst_time.to(device)

        ts = batch["timestamps"]
        if not torch.is_tensor(ts):
            ts = torch.as_tensor(ts, device=device)
        else:
            ts = ts.to(device)

        ld = batch["current_dis"]
        if not torch.is_tensor(ld):
            ld = torch.as_tensor(ld, device=device)
        else:
            ld = ld.to(device)

        current_loc = batch["POI_id"]
        if not torch.is_tensor(current_loc):
            current_loc = torch.as_tensor(current_loc, device=device)
        else:
            current_loc = current_loc.to(device)

        loc_len = batch["mask"]
        if torch.is_tensor(loc_len):
            loc_len_list = loc_len.tolist()
        else:
            loc_len_list = list(loc_len)

        if ts.ndim == 1:
            ts = ts.unsqueeze(1)
        td = dst_time.unsqueeze(1) - ts  # [B, L]
        td_upper = (self.up_time - td).float()
        td_lower = td.float()

        if ld.ndim == 1:
            ld = ld.unsqueeze(1)
        ld_upper = (self.up_loc - ld).float()
        ld_lower = ld.float()

        # hidden
        h_tq = self.forward(
            td_upper, td_lower, ld_upper, ld_lower, current_loc, loc_len_list
        )  # [B,H]
        p_u = self.permanet_weight(user)  # [B,H]
        user_vector = h_tq + p_u  # [B,H]

        # scores over all locations: [B, loc_size]
        ret = torch.mm(user_vector, self.location_weight.weight.T)
        return ret


# ---- Train / Inference entrypoints ----


def train(
    train_dl: DataLoader,
    val_dl: DataLoader,
    view_value: dict[str, Any],
    eval_funcs: dict[str, Callable],
    **kwargs,
) -> Iterable[Sequence[dict[str, Any]]]:
    global model, optimizer
    if model is None:
        data_feature = {
            "loc_size": view_value["loc_size"],
            "uid_size": view_value["uid_size"],
            "tim_size": view_value["tim_size"],
            "distance_upper": view_value["distance_upper"],
        }
        model = STRNN(model_args, data_feature).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_args["learning_rate"])

    for epoch in range(model_args["epochs"]):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_dl):
            optimizer.zero_grad(set_to_none=True)
            loss = model.calculate_loss(batch)
            loss.backward()
            # 可选调试输出：
            # _quick_checks(model, loss)
            # _print_grad_stats(model)
            optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
        avg = total_loss / max(1, n_batches)
        logger.info(f"Epoch {epoch + 1}/{model_args['epochs']}, Loss: {avg:.10f}")

        inference_res = inference(val_dl, view_value)
        pred = inference_res['pred']
        gt = inference_res['gts']
        scores = {}
        for name, func in eval_funcs.items():
            score = func(pred, gt)
            scores[name] = score

        yield [scores, {'loss': avg, 'title':'train_loss'}]

@torch.no_grad()
def inference(
    test_dl: DataLoader, view_value: dict[str, Any], **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    global model, optimizer
    if model is None:
        raise ValueError(
            "Model has not been trained yet. Please train the model before inference."
        )
    model.eval()
    predictions, truth = [], []
    with torch.no_grad():
        for batch in test_dl:
            pred = model.predict(batch)
            predictions.append(pred.cpu())
            truth.append(batch["y_POI_id"]["POI_id"].detach().cpu().numpy().tolist())
    predictions_tensor = torch.cat(predictions, dim=0).cpu().numpy()
    truth = np.concatenate(truth, axis=0)
    return {'pred': predictions_tensor, 'gts': truth}
