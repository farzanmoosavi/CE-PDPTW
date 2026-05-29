import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import pickle
import os
import random
import math
from reward import ALPHA_U as _ALPHA_U_REWARD

INF = float('inf')
SEED_BASE = 42

SCALE_M_PER_COORD = 200.0
UAV_CRUISE_MPS = 20.0
ADR_CRUISE_MPS = 8.3

V_UAV_COORD_PER_MIN = UAV_CRUISE_MPS * 60.0 / SCALE_M_PER_COORD
V_ADR_COORD_PER_MIN = ADR_CRUISE_MPS * 60.0 / SCALE_M_PER_COORD
UAV_LAND_TAKEOFF_MIN = 2.0
V_UAV_MIN_PICKUP = 8.0 * 60.0 / SCALE_M_PER_COORD
V_ADR_MIN_PICKUP = 2.0 * 60.0 / SCALE_M_PER_COORD

def _euclidean_dist_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1)).astype(np.float32)

def _segment_intersects_circle(p1, p2, center, radius):
    d = p2 - p1
    f = p1 - center
    a = float(np.dot(d, d))
    b = float(2 * np.dot(f, d))
    c = float(np.dot(f, f) - radius ** 2)
    disc = b ** 2 - 4 * a * c
    if disc < 0 or a == 0:
        return False
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def _segments_intersect_circle_batch(p1: np.ndarray, p2: np.ndarray,
                                     center: np.ndarray, radius: float) -> np.ndarray:
    d = p2 - p1
    f = p1 - center
    a = (d * d).sum(-1)
    b = 2.0 * (f * d).sum(-1)
    c = (f * f).sum(-1) - radius ** 2
    disc = b ** 2 - 4.0 * a * c
    valid = (disc >= 0) & (a > 1e-12)
    disc_safe = np.where(valid, disc, 0.0)
    disc_sqrt = np.sqrt(disc_safe)
    denom = 2.0 * a + 1e-12
    t1 = np.where(valid, (-b - disc_sqrt) / denom, 2.0)
    t2 = np.where(valid, (-b + disc_sqrt) / denom, 2.0)
    return valid & (((t1 >= 0) & (t1 <= 1)) | ((t2 >= 0) & (t2 <= 1)))

def _segments_intersect(p1, p2, p3, p4):
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)

def vectorized_floyd_warshall(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    for k in range(n):
        dist = np.minimum(dist, dist[:, k, np.newaxis] + dist[k, :])
    return dist

def _synthetic_no_fly_circle(coords: np.ndarray, rng: np.random.Generator):
    cx, cy = float(rng.uniform(0, 5)), float(rng.uniform(0, 5))
    radius = float(rng.uniform(0.4, 0.9))
    dists = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    return dists > radius, cx, cy, radius

def _synthetic_adr_barrier(coords: np.ndarray, rng: np.random.Generator):
    pct_blocked = rng.uniform(0.15, 0.25)
    y_threshold = np.percentile(coords[:, 1], 100.0 * (1.0 - pct_blocked))
    return coords[:, 1] <= y_threshold

def _synthetic_building_clusters(coords: np.ndarray, rng: np.random.Generator):
    n_clusters = int(rng.integers(3, 6))
    clusters = []
    for _ in range(n_clusters):
        cx = float(rng.uniform(0.3, 4.7))
        cy = float(rng.uniform(0.3, 4.7))
        radius = float(rng.uniform(0.2, 0.5))
        clusters.append((np.array([cx, cy], dtype=np.float32), radius))
    return clusters

def static_signed_margin(t_pickup: np.ndarray, dist_matrix: np.ndarray, v_max: float) -> np.ndarray:
    n = len(t_pickup)
    tp_i = t_pickup[:, None].repeat(n, axis=1)
    tp_j = t_pickup[None, :].repeat(n, axis=0)
    return (tp_j - tp_i - dist_matrix / v_max).astype(np.float32)

def create_instance(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                    rng: np.random.Generator, lower=0.0, high=5.0,
                    demand_range=(1.0, 10.0),
                    wind_speed_range=(0.0, 12.0),
                    tw_slack_mean=30.0, tw_slack_std=5.0,
                    tw_slack_clip=(15.0, 60.0),
                    temporal_peaks=False,
                    spatial_cluster_k=0,
                    demand_heavy_tail=False):
    n_depots = n_depots_uav + n_depots_adr
    n_nodes = n_req * 2
    n_total = n_depots + n_nodes

    coords = rng.uniform(lower, high, size=(n_total, 2)).astype(np.float32)

    # Spatial clustering: pull customer nodes toward k cluster centers.
    # Depots stay uniform so agents don't all start near the clusters.
    if spatial_cluster_k > 0:
        k = spatial_cluster_k
        centers = rng.uniform(lower + 0.5, high - 0.5, size=(k, 2)).astype(np.float32)
        sigma = (high - lower) * 0.15
        assigns = rng.integers(0, k, size=n_nodes)
        clustered = (centers[assigns] + rng.normal(0.0, sigma, size=(n_nodes, 2))).astype(np.float32)
        coords[n_depots:] = np.clip(clustered, lower, high)

    depot_uav_xy = coords[:n_depots_uav]
    depot_adr_xy = coords[n_depots_uav:n_depots]
    pickup_xy = coords[n_depots:n_depots + n_req]
    delivery_xy = coords[n_depots + n_req:]

    # Heavy-tail demand: lognormal so a few orders are much larger than the rest.
    if demand_heavy_tail:
        raw = rng.lognormal(mean=0.8, sigma=0.9, size=n_req).astype(np.float32)
        demand_vals = np.clip(raw, demand_range[0], demand_range[1])
    else:
        demand_vals = rng.uniform(demand_range[0], demand_range[1], size=n_req).astype(np.float32)
    demand = np.zeros(n_total, dtype=np.float32)
    demand[n_depots:n_depots + n_req] = demand_vals
    demand[n_depots + n_req:] = -demand_vals

    shift_minutes = 120.0
    if temporal_peaks:
        n_peaks = int(rng.integers(2, 4))
        peak_times = rng.uniform(20.0, 100.0, size=n_peaks)
        n_peak = max(1, int(n_req * 0.4))
        n_bg = n_req - n_peak
        bg = np.sort(rng.uniform(0.0, shift_minutes * 0.9, size=n_bg))
        peak_src = rng.integers(0, n_peaks, size=n_peak)
        peak = np.clip(rng.normal(peak_times[peak_src], 10.0 / 3.0),
                       0.0, shift_minutes - 10.0)
        t_pickup = np.sort(np.concatenate([bg, peak])).astype(np.float32)
    else:
        inter_arrival = rng.exponential(scale=shift_minutes / n_req, size=n_req)
        t_pickup = np.cumsum(inter_arrival).astype(np.float32)
        t_pickup = np.clip(t_pickup, 0, shift_minutes - 10)
    delivery_slack = rng.normal(loc=tw_slack_mean, scale=tw_slack_std, size=n_req).astype(np.float32)
    delivery_slack = np.clip(delivery_slack, tw_slack_clip[0], tw_slack_clip[1])
    t_delivery = (t_pickup + delivery_slack).astype(np.float32)

    acc_uav, nfz_cx, nfz_cy, nfz_radius = _synthetic_no_fly_circle(coords, rng)
    acc_adr = _synthetic_adr_barrier(coords, rng)

    d_adr = _build_road_network(coords, rng)
    d_adr = vectorized_floyd_warshall(d_adr)
    d_uav = _euclidean_dist_matrix(coords)
    nfz_center = np.array([nfz_cx, nfz_cy], dtype=np.float32)
    ii, jj = np.meshgrid(np.arange(n_depots, n_total),
                         np.arange(n_depots, n_total), indexing='ij')
    ii_f, jj_f = ii.flatten(), jj.flatten()
    off_diag = ii_f != jj_f
    ii_f, jj_f = ii_f[off_diag], jj_f[off_diag]
    blocked = _segments_intersect_circle_batch(
        coords[ii_f], coords[jj_f], nfz_center, nfz_radius
    )
    d_uav[ii_f[blocked], jj_f[blocked]] = INF

    for _bc_center, _bc_radius in _synthetic_building_clusters(coords, rng):
        _bld_blocked = _segments_intersect_circle_batch(
            coords[ii_f], coords[jj_f], _bc_center, _bc_radius
        )
        d_uav[ii_f[_bld_blocked], jj_f[_bld_blocked]] = INF

    d_uav = vectorized_floyd_warshall(d_uav)
    for i in range(n_depots, n_total):
        if not acc_uav[i] and not acc_adr[i]:
            acc_adr[i] = True

    Q_uav = 5.0
    for i in range(n_req):
        node_i = n_depots + i
        if acc_uav[node_i] and not acc_adr[node_i]:
            demand_vals[i] = min(demand_vals[i], Q_uav)
            demand[n_depots + i] = demand_vals[i]
            demand[n_depots + n_req + i] = -demand_vals[i]

    wind_dir = float(rng.uniform(0, 2 * math.pi))
    wind_mag = float(rng.uniform(wind_speed_range[0], wind_speed_range[1]))

    from energy import uav_energy_matrix, adr_energy_matrix
    wind_vec = np.array([wind_mag * math.cos(wind_dir), wind_mag * math.sin(wind_dir)], dtype=np.float32)
    Q_adr = 10.0
    e_uav = uav_energy_matrix(
        d_uav * SCALE_M_PER_COORD, demand_vals, demand, n_depots, n_req, wind_vec, UAV_CRUISE_MPS
    )
    e_adr = adr_energy_matrix(
        d_adr * SCALE_M_PER_COORD, demand_vals, demand, n_depots, n_req, ADR_CRUISE_MPS
    )

    tp_full = np.zeros(n_total, dtype=np.float32)
    tp_full[n_depots:n_depots + n_req] = t_pickup
    margin_uav = static_signed_margin(tp_full, d_uav, V_UAV_COORD_PER_MIN)
    margin_adr = static_signed_margin(tp_full, d_adr, V_ADR_COORD_PER_MIN)

    DIST_SCALE = high * math.sqrt(2)
    ENERGY_SCALE = 300_000.0

    tp_i = tp_full[:, None]
    tp_j = tp_full[None, :]
    t_gap = np.abs(tp_i - tp_j) / shift_minutes

    edge_attr_uav = np.stack([
        np.clip(margin_uav.reshape(-1) / shift_minutes, -2.0, 2.0),
        np.clip(d_uav.reshape(-1) / DIST_SCALE, 0.0, 2.0),
        np.clip(e_uav.reshape(-1) / ENERGY_SCALE, 0.0, 1.0),
        t_gap.reshape(-1),
    ], axis=-1).astype(np.float32)

    edge_attr_adr = np.stack([
        np.clip(margin_adr.reshape(-1) / shift_minutes, -2.0, 2.0),
        d_adr.reshape(-1) / DIST_SCALE,
        np.clip(e_adr.reshape(-1) / ENERGY_SCALE, 0.0, 1.0),
        t_gap.reshape(-1),
    ], axis=-1).astype(np.float32)

    node_type = np.zeros((n_total, 3), dtype=np.float32)
    node_type[:n_depots, 0] = 1.0
    node_type[n_depots:n_depots + n_req, 1] = 1.0
    node_type[n_depots + n_req:, 2] = 1.0

    t_pickup_feat = np.zeros(n_total, dtype=np.float32)
    t_pickup_feat[n_depots:n_depots + n_req] = t_pickup
    t_delivery_feat = np.zeros(n_total, dtype=np.float32)
    t_delivery_feat[n_depots + n_req:] = t_delivery

    DEMAND_MAX = 6.0
    delivery_slack_node = np.zeros(n_total, dtype=np.float32)
    delivery_slack_node[n_depots:n_depots + n_req] = delivery_slack / shift_minutes
    delivery_slack_node[n_depots + n_req:]         = delivery_slack / shift_minutes

    COORD_SCALE = high
    TIME_SCALE = shift_minutes + 60.0

    node_feat = np.concatenate([
        coords / COORD_SCALE,
        demand[:, None] / DEMAND_MAX,
        t_pickup_feat[:, None] / TIME_SCALE,
        t_delivery_feat[:, None] / TIME_SCALE,
        node_type,
        acc_uav[:, None].astype(np.float32),
        acc_adr[:, None].astype(np.float32),
        delivery_slack_node[:, None],
    ], axis=-1).astype(np.float32)

    mask_adj_uav = (d_uav < 1e9).astype(np.float32) * acc_uav[:, None].astype(np.float32)
    mask_adj_adr = (d_adr < 1e9).astype(np.float32) * acc_adr[:, None].astype(np.float32)

    edges_index = np.stack(np.indices((n_total, n_total)).reshape(2, -1), axis=0)

    capacity = np.hstack([
        np.ones(n_uav, dtype=np.float32) * Q_uav,
        np.ones(n_adr, dtype=np.float32) * Q_adr,
    ])
    BATTERY_UAV = 6500.0
    BATTERY_ADR = 4500.0
    battery = np.hstack([
        np.ones(n_uav, dtype=np.float32) * BATTERY_UAV,
        np.ones(n_adr, dtype=np.float32) * BATTERY_ADR,
    ])

    prep_time = rng.uniform(5.0, 20.0, size=n_req).astype(np.float32)
    t_arrival = np.maximum(0.0, t_pickup - prep_time)

    return {
        'x': torch.tensor(node_feat, dtype=torch.float32),
        'edge_index': torch.tensor(edges_index, dtype=torch.long),
        'edge_attr_uav': torch.tensor(edge_attr_uav, dtype=torch.float32),
        'edge_attr_adr': torch.tensor(edge_attr_adr, dtype=torch.float32),
        'mask_adjacency_uav': torch.tensor(mask_adj_uav.reshape(-1, 1), dtype=torch.float32),
        'mask_adjacency_adr': torch.tensor(mask_adj_adr.reshape(-1, 1), dtype=torch.float32),
        'demand': torch.tensor(demand, dtype=torch.float32).unsqueeze(-1),
        'capacity': torch.tensor(capacity, dtype=torch.float32),
        'battery': torch.tensor(battery, dtype=torch.float32),
        'edge_attr_d': torch.tensor(d_uav.reshape(-1, 1), dtype=torch.float32),
        'edge_attr_r': torch.tensor(d_adr.reshape(-1, 1), dtype=torch.float32),
        'time_window': torch.tensor(
            np.concatenate([np.zeros(n_depots), t_pickup, t_delivery])[:, None],
            dtype=torch.float32
        ),
        'wind': torch.tensor([wind_mag, wind_dir], dtype=torch.float32),
        'n_depots': n_depots,
        'n_depots_uav': n_depots_uav,
        'n_req': n_req,
        't_arrival': torch.tensor(t_arrival, dtype=torch.float32),
    }

def _build_road_network(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(coords)
    dist = np.full((n, n), INF, dtype=np.float32)
    np.fill_diagonal(dist, 0)
    for i in range(n - 1):
        n_neighbors = max(1, (n - i - 1) // 2)
        neighbors = rng.choice(range(i + 1, n), size=n_neighbors, replace=False)
        dists_ij = np.linalg.norm(coords[i] - coords[neighbors], axis=1)
        dist[i, neighbors] = dists_ij
        dist[neighbors, i] = dists_ij
    return dist

class CEPDPTWDataset(Dataset):
    def __init__(self, n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                 num_samples, seed=None,
                 tw_slack_mean=30.0, tw_slack_std=5.0, tw_slack_clip=(15.0, 60.0),
                 temporal_peaks=False, spatial_cluster_k=0, demand_heavy_tail=False):
        self.n_req = n_req
        self.n_uav = n_uav
        self.n_adr = n_adr
        self.n_depots_uav = n_depots_uav
        self.n_depots_adr = n_depots_adr
        self.num_samples = num_samples
        base_seed = seed if seed is not None else SEED_BASE
        self.data = [
            create_instance(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                            np.random.default_rng(base_seed + i),
                            tw_slack_mean=tw_slack_mean,
                            tw_slack_std=tw_slack_std,
                            tw_slack_clip=tw_slack_clip,
                            temporal_peaks=temporal_peaks,
                            spatial_cluster_k=spatial_cluster_k,
                            demand_heavy_tail=demand_heavy_tail)
            for i in range(num_samples)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CEPDPTWStreamDataset(Dataset):
    def __init__(self, n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                 num_samples, seed=None,
                 tw_slack_mean=30.0, tw_slack_std=5.0, tw_slack_clip=(15.0, 60.0),
                 temporal_peaks=False, spatial_cluster_k=0, demand_heavy_tail=False,
                 mix_hard_frac=0.0):
        self.n_req = n_req
        self.n_uav = n_uav
        self.n_adr = n_adr
        self.n_depots_uav = n_depots_uav
        self.n_depots_adr = n_depots_adr
        self.num_samples = num_samples
        self.base_seed = seed if seed is not None else SEED_BASE
        self.tw_slack_mean     = tw_slack_mean
        self.tw_slack_std      = tw_slack_std
        self.tw_slack_clip     = tw_slack_clip
        self.temporal_peaks    = temporal_peaks
        self.spatial_cluster_k = spatial_cluster_k
        self.demand_heavy_tail = demand_heavy_tail
        self.mix_hard_frac     = mix_hard_frac

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Use a separate RNG for sampling instance flags so the instance-generation
        # RNG (base_seed + idx) is unaffected whether mix_hard_frac is on or off.
        rng_flags = np.random.default_rng(self.base_seed + idx + 1_000_000_000)
        rng_inst  = np.random.default_rng(self.base_seed + idx)

        if self.mix_hard_frac > 0.0:
            f = self.mix_hard_frac
            tp  = bool(rng_flags.random() < f)
            sck = int(rng_flags.integers(2, 5)) if rng_flags.random() < f else 0
            dht = bool(rng_flags.random() < f)
            # Tighten TW proportionally to how many hard flags fired
            n_hard = int(tp) + int(sck > 0) + int(dht)
            if n_hard > 0:
                blend = n_hard / 3.0
                tw_mean = self.tw_slack_mean * (1.0 - blend) + 10.0 * blend
                tw_clip_lo = self.tw_slack_clip[0] * (1.0 - blend) + 5.0 * blend
                tw_clip_hi = self.tw_slack_clip[1] * (1.0 - blend) + 20.0 * blend
            else:
                tw_mean   = self.tw_slack_mean
                tw_clip_lo, tw_clip_hi = self.tw_slack_clip[0], self.tw_slack_clip[1]
        else:
            tp, sck, dht = self.temporal_peaks, self.spatial_cluster_k, self.demand_heavy_tail
            tw_mean   = self.tw_slack_mean
            tw_clip_lo, tw_clip_hi = self.tw_slack_clip[0], self.tw_slack_clip[1]

        return create_instance(self.n_req, self.n_uav, self.n_adr,
                               self.n_depots_uav, self.n_depots_adr, rng_inst,
                               tw_slack_mean=tw_mean,
                               tw_slack_std=self.tw_slack_std,
                               tw_slack_clip=(tw_clip_lo, tw_clip_hi),
                               temporal_peaks=tp,
                               spatial_cluster_k=sck,
                               demand_heavy_tail=dht)

def creat_data(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
               num_samples, batch_size=512, num_workers=0,
               use_cache=False, cache_file='dataset_cache.pkl', seed=None,
               shuffle=None, streaming=False,
               distributed=False, rank=0, world_size=1,
               tw_slack_mean=30.0, tw_slack_std=5.0, tw_slack_clip=(15.0, 60.0),
               temporal_peaks=False, spatial_cluster_k=0, demand_heavy_tail=False,
               mix_hard_frac=0.0):
    _tw_kwargs = dict(tw_slack_mean=tw_slack_mean, tw_slack_std=tw_slack_std,
                      tw_slack_clip=tw_slack_clip,
                      temporal_peaks=temporal_peaks,
                      spatial_cluster_k=spatial_cluster_k,
                      demand_heavy_tail=demand_heavy_tail)
    if streaming:
        dataset = CEPDPTWStreamDataset(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                                       num_samples, seed=seed,
                                       mix_hard_frac=mix_hard_frac, **_tw_kwargs)
    elif use_cache:
        try:
            with open(cache_file, 'rb') as f:
                dataset = pickle.load(f)
            if rank == 0:
                print('Loaded dataset from cache.')
        except (FileNotFoundError, EOFError):
            if rank == 0 and os.path.exists(cache_file):
                print(f'Cache {cache_file} corrupted — deleting and regenerating.')
                os.remove(cache_file)
            dataset = CEPDPTWDataset(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                                     num_samples, seed=seed, **_tw_kwargs)
            if rank == 0:
                with open(cache_file, 'wb') as f:
                    pickle.dump(dataset, f)
                print('Saved dataset to cache.')
    else:
        dataset = CEPDPTWDataset(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                                 num_samples, seed=seed, **_tw_kwargs)

    if distributed and world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=(shuffle is not False))
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                          persistent_workers=(num_workers > 0))

    if shuffle is None:
        if streaming:
            shuffle = True
        else:
            cache_name = str(cache_file).lower()
            shuffle = ('valid' not in cache_name) and ('test' not in cache_name)

    pin = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin,
                      persistent_workers=(num_workers > 0))

def build_dynamic_sub_batch(batch: dict, n_vis: int, n_req: int, n_depots: int,
                            t_current=None, dynamic_features=False) -> dict:
    """
    Slice a full batch to expose only revealed requests.

    If t_current is provided and 't_arrival' is in batch: n_vis is determined by
    how many requests have t_arrival <= t_current (min across the batch), giving
    training-evaluation alignment with the rolling-horizon dispatcher.
    Otherwise falls back to the supplied n_vis cap (random index-based slicing).

    If dynamic_features=True and t_current is provided: augments node features
    with order_age (at pickup nodes) and delivery_slack_dynamic (at delivery nodes),
    expanding x from 11D to 13D.  Requires model trained with input_node_dim=13.
    """
    device = batch['x'].device
    B      = batch['x'].shape[0]

    # ── Determine n_vis from t_arrival when t_current is given ──────────────
    if t_current is not None and 't_arrival' in batch:
        t_arr = batch['t_arrival']          # [B, n_req]
        n_vis = int((t_arr <= t_current).sum(dim=1).min().item())
        n_vis = max(1, min(n_vis, n_req))

    n_total = n_depots + n_req * 2
    n_sub   = n_depots + n_vis * 2

    keep = (
        list(range(n_depots)) +
        list(range(n_depots, n_depots + n_vis)) +
        list(range(n_depots + n_req, n_depots + n_req + n_vis))
    )
    keep_t = torch.tensor(keep, dtype=torch.long, device=device)

    def _slice_edge(ea):
        D   = ea.shape[-1]
        mat = ea.view(B, n_total, n_total, D)
        return mat[:, keep_t][:, :, keep_t].reshape(B, n_sub * n_sub, D)

    idx = torch.arange(n_sub, device=device)
    ii, jj = torch.meshgrid(idx, idx, indexing='ij')
    ei = torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=0)

    sub = {
        'x':                  batch['x'][:, keep_t],
        'demand':             batch['demand'][:, keep_t],
        'time_window':        batch['time_window'][:, keep_t],
        'capacity':           batch['capacity'],
        'battery':            batch['battery'],
        'wind':               batch['wind'],
        'n_depots':           n_depots,
        'n_depots_uav':       batch.get('n_depots_uav', n_depots),
        'n_req':              n_vis,
        'edge_attr_uav':      _slice_edge(batch['edge_attr_uav']),
        'edge_attr_adr':      _slice_edge(batch['edge_attr_adr']),
        'edge_attr_d':        _slice_edge(batch['edge_attr_d']),
        'edge_attr_r':        _slice_edge(batch['edge_attr_r']),
        'mask_adjacency_uav': _slice_edge(batch['mask_adjacency_uav']),
        'mask_adjacency_adr': _slice_edge(batch['mask_adjacency_adr']),
        'edge_index':         ei.unsqueeze(0).expand(B, -1, -1).contiguous(),
    }
    if 't_arrival' in batch:
        sub['t_arrival'] = batch['t_arrival'][:, :n_vis]

    # ── Dynamic node features: order_age + delivery_slack_dynamic ───────────
    if dynamic_features and t_current is not None:
        _TIME_SCALE = 180.0
        if 't_arrival' in batch:
            t_arr_vis = batch['t_arrival'][:, :n_vis]           # [B, n_vis]
            order_age = ((t_current - t_arr_vis) / _TIME_SCALE).clamp(0.0, 1.0)
        else:
            order_age = sub['x'].new_zeros(B, n_vis)
        # delivery time windows for revealed requests: positions n_depots+n_vis to n_sub
        t_del = sub['time_window'][:, n_depots + n_vis:n_sub, 0]  # [B, n_vis]
        del_slack = ((t_del - t_current) / _TIME_SCALE).clamp(-1.0, 1.0)
        extra = sub['x'].new_zeros(B, n_sub, 2)
        extra[:, n_depots:n_depots + n_vis, 0] = order_age
        extra[:, n_depots + n_vis:n_sub, 1]    = del_slack
        sub['x'] = torch.cat([sub['x'], extra], dim=-1)  # [B, n_sub, 13]

    return sub


def build_dynamic_masked_batch(batch: dict, n_req: int, n_depots: int,
                               t_current: float, dynamic_features: bool = True) -> tuple:
    """
    Full-graph dynamic masking without slicing.
    Returns (masked_batch, initial_visited, vis_counts).

    Zeroes x / time_window / demand for unrevealed nodes per-instance and
    optionally appends order_age + delivery_slack_dynamic (→ 13-D x).
    initial_visited is [B, n_total] with 1 for every unrevealed node.
    vis_counts is a [B] long tensor with the number of revealed orders per instance.
    """
    device = batch['x'].device
    B      = batch['x'].shape[0]
    n_total = n_depots + n_req * 2

    # Per-instance revealed mask
    if 't_arrival' in batch:
        t_arr    = batch['t_arrival']                               # [B, n_req]
        revealed = (t_arr <= t_current)                             # [B, n_req] bool
        vis_counts = revealed.sum(dim=1).clamp(min=1, max=n_req)    # [B] long
    else:
        revealed   = torch.ones(B, n_req, dtype=torch.bool, device=device)
        vis_counts = torch.full((B,), n_req, dtype=torch.long, device=device)

    unrevealed = ~revealed  # [B, n_req] bool

    # Per-instance initial_visited: [B, n_total]
    initial_visited = batch['x'].new_zeros(B, n_total)
    initial_visited[:, n_depots:n_depots + n_req]           = unrevealed.float()
    initial_visited[:, n_depots + n_req:n_depots + 2*n_req] = unrevealed.float()

    out = dict(batch)

    # Zero unrevealed nodes per-instance
    revealed_f = revealed.float().unsqueeze(-1)  # [B, n_req, 1]

    x = batch['x'].clone()
    x[:, n_depots:n_depots + n_req, :]           *= revealed_f
    x[:, n_depots + n_req:n_depots + 2*n_req, :] *= revealed_f
    out['x'] = x

    if 'time_window' in batch:
        tw = batch['time_window'].clone()
        tw[:, n_depots:n_depots + n_req, :]           *= revealed_f
        tw[:, n_depots + n_req:n_depots + 2*n_req, :] *= revealed_f
        out['time_window'] = tw

    if 'demand' in batch:
        dm = batch['demand'].clone()
        dm[:, n_depots:n_depots + n_req, :]           *= revealed_f
        dm[:, n_depots + n_req:n_depots + 2*n_req, :] *= revealed_f
        out['demand'] = dm

    # Dynamic node features: order_age (pickup) + delivery_slack (delivery)
    if dynamic_features:
        _TIME_SCALE = 180.0
        x_base = out['x']
        extra  = x_base.new_zeros(B, n_total, 2)
        if 't_arrival' in batch:
            order_age = ((t_current - batch['t_arrival']) / _TIME_SCALE).clamp(0.0, 1.0)
            order_age = order_age * revealed.float()              # zero unrevealed
            extra[:, n_depots:n_depots + n_req, 0] = order_age
        t_del     = out['time_window'][:, n_depots + n_req:n_depots + 2*n_req, 0]
        del_slack = ((t_del - t_current) / _TIME_SCALE).clamp(-1.0, 1.0)
        del_slack = del_slack * revealed.float()                  # zero unrevealed
        extra[:, n_depots + n_req:n_depots + 2*n_req, 1] = del_slack
        out['x'] = torch.cat([x_base, extra], dim=-1)            # [B, n_total, 13]

    return out, initial_visited, vis_counts


def _pickup_speed(distance, t_now, t_target, v_min, v_max):
    slack = torch.clamp(t_target - t_now, min=0.0)
    exact_speed = distance / torch.clamp(slack, min=1e-8)
    speed = torch.clamp(exact_speed, min=v_min, max=v_max)
    must_rush = (slack <= 0.0) | (distance / v_max >= slack)
    speed = torch.where(must_rush, torch.full_like(speed, v_max), speed)
    speed = torch.where(distance <= 1e-8, torch.full_like(speed, v_min), speed)
    return speed

def reward1(time_window, tour_indices, edge_attr_d, edge_attr_r, time, num_drones,
            num_depots=None, return_breakdown=False):
    dev = tour_indices.device
    batch_size, n_agent, steps = tour_indices.size()
    if num_depots is None:
        num_depots = int((time_window == 0).sum().item() // batch_size)
    num_nodes = time_window.size(1) - num_depots
    num_pickup = num_nodes // 2

    tw = time_window.view(batch_size, num_depots + num_nodes)
    edge_attr_d = edge_attr_d.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)
    edge_attr_r = edge_attr_r.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)

    prev_idx = tour_indices[:, :, :-1]
    cur_idx = tour_indices[:, :, 1:]

    is_depot = cur_idx < num_depots
    is_pickup = (cur_idx >= num_depots) & (cur_idx < num_depots + num_pickup)
    is_delivery = cur_idx >= num_depots + num_pickup
    is_customer = is_pickup | is_delivery
    is_consec_depot = is_depot & (prev_idx < num_depots)

    ALPHA_1 = 0.60
    ALPHA_2 = 0.10
    ALPHA_E = 0.02
    ALPHA_P = 0.10
    ALPHA_D = 0.15
    ALPHA_U = _ALPHA_U_REWARD

    n_uav = num_drones
    batch_d = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, n_uav).unsqueeze(2)
    batch_r = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, n_agent - n_uav).unsqueeze(2)

    dis_d = edge_attr_d[batch_d, prev_idx[:, :n_uav], cur_idx[:, :n_uav]]
    dis_r = edge_attr_r[batch_r, prev_idx[:, n_uav:], cur_idx[:, n_uav:]]

    prev_time = torch.cat(
        [torch.zeros(batch_size, n_agent, 1, device=dev), time[:, :, :-1]],
        dim=2,
    )
    tw_cur = torch.gather(tw.unsqueeze(1).expand(-1, n_agent, -1), 2, cur_idx)

    travel_time_uav = dis_d / V_UAV_COORD_PER_MIN
    travel_time_adr = dis_r / V_ADR_COORD_PER_MIN

    if n_uav > 0:
        pickup_uav = is_pickup[:, :n_uav]
        if pickup_uav.any():
            travel_time_uav = travel_time_uav.clone()
            speed_u = _pickup_speed(
                dis_d,
                prev_time[:, :n_uav, :],
                tw_cur[:, :n_uav, :],
                v_min=V_UAV_MIN_PICKUP,
                v_max=V_UAV_COORD_PER_MIN,
            )
            travel_time_uav[pickup_uav] = dis_d[pickup_uav] / speed_u[pickup_uav]

    if n_agent - n_uav > 0:
        pickup_adr = is_pickup[:, n_uav:]
        if pickup_adr.any():
            travel_time_adr = travel_time_adr.clone()
            speed_a = _pickup_speed(
                dis_r,
                prev_time[:, n_uav:, :],
                tw_cur[:, n_uav:, :],
                v_min=V_ADR_MIN_PICKUP,
                v_max=V_ADR_COORD_PER_MIN,
            )
            travel_time_adr[pickup_adr] = dis_r[pickup_adr] / speed_a[pickup_adr]

    travel_time = torch.cat([travel_time_uav, travel_time_adr], dim=1)
    travel_time = torch.where(is_consec_depot, torch.zeros_like(travel_time), travel_time)

    alpha_mode = torch.cat([
        torch.full((batch_size, n_uav, steps - 1), ALPHA_1, device=dev),
        torch.full((batch_size, n_agent - n_uav, steps - 1), ALPHA_2, device=dev),
    ], dim=1)
    operating_cost = (alpha_mode * travel_time).sum(dim=2)

    arrival_pre_service = prev_time + travel_time

    completion_time = arrival_pre_service.clone()
    if n_uav > 0:
        uav_customer = is_customer[:, :n_uav].float()
        completion_time[:, :n_uav, :] = completion_time[:, :n_uav, :] + UAV_LAND_TAKEOFF_MIN * uav_customer

    early_pickup = F.relu(tw_cur - arrival_pre_service) * is_pickup.float()
    late_pickup = F.relu(arrival_pre_service - tw_cur) * is_pickup.float()
    late_delivery = F.relu(completion_time - tw_cur) * is_delivery.float()

    early_pickup = torch.where(is_consec_depot, torch.zeros_like(early_pickup), early_pickup)
    late_pickup = torch.where(is_consec_depot, torch.zeros_like(late_pickup), late_pickup)
    late_delivery = torch.where(is_consec_depot, torch.zeros_like(late_delivery), late_delivery)

    delivery_start = num_depots + num_pickup
    visited_flat = tour_indices.reshape(batch_size, -1)
    delivery_nodes = torch.arange(delivery_start, delivery_start + num_pickup, device=dev)
    visited_any = (visited_flat.unsqueeze(2) == delivery_nodes).any(dim=1)
    n_undeliv = (~visited_any).sum(dim=1).float()

    undeliv_cost = (ALPHA_U * n_undeliv / n_agent).unsqueeze(1).expand(-1, n_agent)

    tw_pen = (
        ALPHA_P * late_pickup.sum(dim=2)
        + ALPHA_D * late_delivery.sum(dim=2)
        + ALPHA_E * early_pickup.sum(dim=2)
    )
    total_cost = operating_cost + tw_pen + undeliv_cost
    if return_breakdown:
        return total_cost, {
            'travel':     operating_cost,
            'tw_penalty': tw_pen,
            'undeliv':    undeliv_cost,
        }
    return total_cost

