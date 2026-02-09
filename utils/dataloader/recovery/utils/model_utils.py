import torch
from utils.mbr import MBR
from utils.spatial_func import LAT_PER_METER, LNG_PER_METER
from utils.spatial_func import SPoint
from utils.graph_func import empty_graph
import dgl


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_online_info_dict(
    grid_rn_dict, norm_grid_poi_dict, norm_grid_rnfea_dict, parameters
):
    rid_grid_dict = get_rid_grid_dict(grid_rn_dict)
    online_features_dict = {}
    for rid in rid_grid_dict.keys():
        online_feas = []
        for grid in rid_grid_dict[rid]:
            try:
                poi = norm_grid_poi_dict[grid]
            except:
                poi = [0.0] * 5
            try:
                rnfea = norm_grid_rnfea_dict[grid]
            except:
                rnfea = [0.0] * 5
            online_feas.append(poi + rnfea)

        online_feas = np.array(online_feas)
        online_features_dict[rid] = list(online_feas.mean(axis=0))

    online_features_dict[0] = [0.0] * online_feas.shape[1]  # add soss

    return online_features_dict

def get_rid_rnfea_dict(rn_dict, parameters):
    df = pd.DataFrame(rn_dict).T

    # standardization length
    df["norm_len"] = [np.log10(l) / np.log10(df["length"].max()) for l in df["length"]]
    #         df['norm_len'] = (df['length'] - df['length'].mean())/df['length'].std()

    # one hot road level
    one_hot_df = pd.get_dummies(df.level, prefix="level")
    df = df.join(one_hot_df)

    # get number of neighbours (standardization)
    g = nx.Graph()
    edges = []
    for coords in df["coords"].values:
        start_node = (coords[0].lat, coords[0].lng)
        end_node = (coords[-1].lat, coords[-1].lng)
        edges.append((start_node, end_node))
    g.add_edges_from(edges)

    num_start_neighbors = []
    num_end_neighbors = []
    for coords in df["coords"].values:
        start_node = (coords[0].lat, coords[0].lng)
        end_node = (coords[-1].lat, coords[-1].lng)
        num_start_neighbors.append(len(list(g.edges(start_node))))
        num_end_neighbors.append(len(list(g.edges(end_node))))
    df["num_start_neighbors"] = num_start_neighbors
    df["num_end_neighbors"] = num_end_neighbors
    start = df["num_start_neighbors"]
    end = df["num_end_neighbors"]
    # distribution is like gaussian --> use min max normalization
    df["norm_num_start_neighbors"] = (start - start.min()) / (start.max() - start.min())
    df["norm_num_end_neighbors"] = (end - end.min()) / (end.max() - end.min())

    # convert to dict <key:rid, value:fea>
    norm_rid_rnfea_dict = {}
    for i in range(len(df)):
        k = df.index[i]
        v = df.iloc[i][
            [
                "norm_len",
                "level_2",
                "level_3",
                "level_4",
                "level_5",
                "level_6",
                "norm_num_start_neighbors",
                "norm_num_end_neighbors",
            ]
        ]
        norm_rid_rnfea_dict[k] = list(v)

    norm_rid_rnfea_dict[0] = [0.0] * len(list(v))  # add soss
    return norm_rid_rnfea_dict


def get_dis_prob_vec(gps, rn, parameters, search_dist=None, beta=None):
    """
    Args:
    -----
    gps: [SPoint, tid]
    """
    if search_dist is None:
        search_dist = parameters.search_dist
    if beta is None:
        beta = parameters.beta
    cons_vec = torch.zeros(parameters.id_size)
    mbr = MBR(
        gps[0].lat - search_dist * LAT_PER_METER,
        gps[0].lng - search_dist * LNG_PER_METER,
        gps[0].lat + search_dist * LAT_PER_METER,
        gps[0].lng + search_dist * LNG_PER_METER,
    )
    candis = rn.get_candidates(gps[0], mbr)
    if candis is not None:
        for candi_pt in candis:
            new_rid = rn.valid_edge_one[candi_pt.eid]
            cons_vec[new_rid] = exp_prob(beta, candi_pt.error)
    else:
        cons_vec = torch.ones(parameters.id_size)
    return cons_vec


import math

def get_dict_info_batch(input_id, features_dict):
    """
    batched dict info
    """
    # input_id = [1, batch size]
    input_id = input_id.reshape(-1)
    features = torch.index_select(features_dict, dim=0, index=input_id)
    return features


def exp_prob(beta, x):
    """
    error distance weight.
    """
    return math.exp(-pow(x, 2) / pow(beta, 2))


def get_reachable_inds(parameters):
    reachable_inds = list(range(parameters.id_size))

    return reachable_inds


def get_constraint_mask(
    src_grid_seqs, src_gps_seqs, src_lengths, trg_lengths, rn, parameters
):
    max_trg_len = max(trg_lengths)
    max_src_len = max(src_lengths)
    batch_size = src_grid_seqs.size(0)
    constraint_mat_trg = torch.zeros(batch_size, max_trg_len, parameters.id_size) + 1e-6
    constraint_mat_src = torch.zeros(batch_size, max_src_len, parameters.id_size)

    for bs in range(batch_size):
        # first src gps
        pre_t = 1
        pre_gps = [
            SPoint(
                src_gps_seqs[bs][pre_t][0].tolist(), src_gps_seqs[bs][pre_t][1].tolist()
            ),
            pre_t,
        ]

        if parameters.dis_prob_mask_flag:
            constraint_mat_src[bs][pre_t] = get_dis_prob_vec(
                pre_gps, rn, parameters, parameters.neighbor_dist, parameters.gamma
            )
            constraint_mat_trg[bs][pre_t] = get_dis_prob_vec(pre_gps, rn, parameters)
        else:
            reachable_inds = get_reachable_inds(parameters)
            constraint_mat_trg[bs][pre_t][reachable_inds] = 1

        # missed gps
        for i in range(2, src_lengths[bs]):
            cur_t = int(src_grid_seqs[bs, i, 2].tolist())
            cur_gps = [
                SPoint(
                    src_gps_seqs[bs][i][0].tolist(), src_gps_seqs[bs][i][1].tolist()
                ),
                cur_t,
            ]

            time_diff = cur_t - pre_t
            reachable_inds = get_reachable_inds(parameters)

            for t in range(pre_t + 1, cur_t):
                constraint_mat_trg[bs][t][reachable_inds] = 1

            # middle src gps
            if parameters.dis_prob_mask_flag:
                constraint_mat_src[bs][i] = get_dis_prob_vec(
                    cur_gps, rn, parameters, parameters.neighbor_dist, parameters.gamma
                )
                constraint_mat_trg[bs][cur_t] = get_dis_prob_vec(
                    cur_gps, rn, parameters
                )
            else:
                reachable_inds = get_reachable_inds(parameters)
                constraint_mat_trg[bs][cur_t][reachable_inds] = 1
            pre_t = cur_t

    constraint_mat_trg = torch.clip(constraint_mat_trg, 1e-6, 1)
    return constraint_mat_trg, constraint_mat_src


def get_gps_subgraph(constraint_mat_src, src_grid_seq, trg_rid, parameters):
    total_g = parameters.g
    gps_subgraph = [empty_graph()]
    for i in range(1, constraint_mat_src.size(0)):
        sub = dgl.DGLGraph()
        nodes = torch.where(constraint_mat_src[i] > 0)[0].numpy().tolist()
        if trg_rid[src_grid_seq[i][-1]] not in nodes:
            nodes.append(trg_rid[src_grid_seq[i][-1]].item())
        _, neighbor = total_g.out_edges(nodes)
        nodes = list(set.union(set(nodes), set(neighbor.numpy().tolist())))
        sub.add_nodes(len(nodes))
        sub.ndata["id"] = torch.tensor(nodes)
        nmap = {}
        for k, rid in enumerate(nodes):
            nmap[rid] = k
        src, dst, w = [], [], []
        for rid in nodes:
            w.append(constraint_mat_src[i][rid])
            _, neighbor = total_g.out_edges([rid])
            for nrid in neighbor:
                if nrid.item() in nmap:
                    if rid != nrid.item():
                        src.append(nmap[rid])
                        dst.append(nmap[nrid.item()])
        sub.add_edges(src, dst)
        # sub.ndata['w'] = torch.tensor(w).reshape(-1, 1) / sum(w)
        sub.ndata["w"] = torch.tensor(w).reshape(-1, 1)
        sub.ndata["gt"] = torch.zeros_like(sub.ndata["w"])
        sub.ndata["gt"][nmap[trg_rid[src_grid_seq[i][-1]].item()], :] = 1
        sub = dgl.add_self_loop(sub)
        gps_subgraph.append(sub)
    return gps_subgraph
