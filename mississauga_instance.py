from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from creat_vrp import (
    SCALE_M_PER_COORD,
    UAV_CRUISE_MPS,
    ADR_CRUISE_MPS,
    V_UAV_COORD_PER_MIN,
    V_ADR_COORD_PER_MIN,
    static_signed_margin,
)

MISSISSAUGA_POLYGON_LONLAT = [
    (-79.619, 43.476),
    (-79.534, 43.576),
    (-79.620, 43.639),
    (-79.720, 43.548),
]

UAV_NETWORK_FILTER = '["highway"~"primary|secondary"]'
ADR_NETWORK_FILTER = '["highway"~"primary|secondary|residential"]'

PEARSON_LONLAT = (-79.6306, 43.6777)
PEARSON_NO_FLY_RADIUS_KM = 5.6

@dataclass
class MississaugaNetworks:
    coords_km: np.ndarray
    is_restaurant_candidate: np.ndarray
    acc_uav_node: np.ndarray
    acc_adr_node: np.ndarray
    d_uav_km: np.ndarray
    d_adr_km: np.ndarray
    bbox_xy_km: Tuple[float, float, float, float]
    bbox_lonlat: Tuple[float, float, float, float]
    pearson_xy_km: np.ndarray
    cache_key: str

_NETWORKS_CACHE: Optional[MississaugaNetworks] = None
_CACHE_PATH_DEFAULT = "cache/mississauga_networks.pkl"

def _project_lonlat_to_local_km(lon: np.ndarray, lat: np.ndarray,
                                 lon0: float, lat0: float) -> np.ndarray:
    R = 6371.0088
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * R
    y = np.radians(lat - lat0) * R
    return np.stack([x, y], axis=-1).astype(np.float32)

def _build_networks_from_osm(cache_path: str) -> MississaugaNetworks:
    import networkx as nx
    import osmnx as ox
    from shapely.geometry import Polygon

    poly = Polygon(MISSISSAUGA_POLYGON_LONLAT)

    G_uav = ox.graph_from_polygon(poly, simplify=True, retain_all=False,
                                   network_type="all",
                                   custom_filter=UAV_NETWORK_FILTER)
    G_adr = ox.graph_from_polygon(poly, simplify=True, retain_all=False,
                                   network_type="all",
                                   custom_filter=ADR_NETWORK_FILTER)

    components = list(nx.connected_components(G_adr.to_undirected()))
    if len(components) > 1:
        for i in range(len(components) - 1):
            ca = list(components[i])
            cb = list(components[i + 1])
            best = None
            for na in ca[:30]:
                for nb in cb[:30]:
                    d = math.hypot(G_adr.nodes[na]['x'] - G_adr.nodes[nb]['x'],
                                   G_adr.nodes[na]['y'] - G_adr.nodes[nb]['y'])
                    if best is None or d < best[0]:
                        best = (d, na, nb)
            if best is not None:
                _, na, nb = best
                G_adr.add_edge(na, nb, length=best[0] * 111_000)

    candidate_lonlat: Dict[int, Tuple[float, float]] = {}
    for nid, data in G_uav.nodes(data=True):
        candidate_lonlat[nid] = (data['x'], data['y'])
    for nid, data in G_adr.nodes(data=True):
        candidate_lonlat.setdefault(nid, (data['x'], data['y']))

    node_ids = list(candidate_lonlat.keys())
    lon = np.array([candidate_lonlat[n][0] for n in node_ids], dtype=np.float32)
    lat = np.array([candidate_lonlat[n][1] for n in node_ids], dtype=np.float32)
    lon0 = lon.mean()
    lat0 = lat.mean()
    coords_km = _project_lonlat_to_local_km(lon, lat, lon0, lat0)

    coords_km -= coords_km.min(axis=0)
    bbox_xy_km = (0.0, 0.0,
                  float(coords_km[:, 0].max()),
                  float(coords_km[:, 1].max()))
    bbox_lonlat = (float(lon.min()), float(lat.min()),
                   float(lon.max()), float(lat.max()))

    pearson_xy = _project_lonlat_to_local_km(
        np.array([PEARSON_LONLAT[0]]), np.array([PEARSON_LONLAT[1]]),
        lon0, lat0,
    )[0]
    pearson_xy = pearson_xy - np.array(
        _project_lonlat_to_local_km(lon, lat, lon0, lat0).min(axis=0)
    )
    dist_to_pearson = np.linalg.norm(coords_km - pearson_xy, axis=1)
    acc_uav_node = dist_to_pearson > PEARSON_NO_FLY_RADIUS_KM

    adr_node_set = set(G_adr.nodes())
    acc_adr_node = np.array([nid in adr_node_set for nid in node_ids], dtype=bool)

    uav_node_set = set(G_uav.nodes())
    is_restaurant = np.array([nid in uav_node_set for nid in node_ids], dtype=bool)

    print(f"[mississauga] Computing ADR shortest paths on {len(node_ids)} nodes...")
    d_adr_km = _all_pairs_shortest_path_km(G_adr, node_ids, default_blocked=True)

    diff = coords_km[:, None, :] - coords_km[None, :, :]
    d_uav_km = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)
    _block_segments_through_circle(d_uav_km, coords_km,
                                    pearson_xy, PEARSON_NO_FLY_RADIUS_KM)

    cache_key = (f"poly={MISSISSAUGA_POLYGON_LONLAT}|"
                 f"uav={UAV_NETWORK_FILTER}|adr={ADR_NETWORK_FILTER}|"
                 f"pearson={PEARSON_LONLAT}|r={PEARSON_NO_FLY_RADIUS_KM}")

    nets = MississaugaNetworks(
        coords_km=coords_km,
        is_restaurant_candidate=is_restaurant,
        acc_uav_node=acc_uav_node,
        acc_adr_node=acc_adr_node,
        d_uav_km=d_uav_km,
        d_adr_km=d_adr_km,
        bbox_xy_km=bbox_xy_km,
        bbox_lonlat=bbox_lonlat,
        pearson_xy_km=pearson_xy.astype(np.float32),
        cache_key=cache_key,
    )

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(nets, f)
    print(f"[mississauga] Saved network cache to {cache_path}")
    return nets

def _all_pairs_shortest_path_km(G, node_ids: List[int],
                                  default_blocked: bool = True) -> np.ndarray:
    import networkx as nx

    n = len(node_ids)
    INF = float("inf")
    out = np.full((n, n), INF if default_blocked else 0.0, dtype=np.float32)
    idx_of = {nid: i for i, nid in enumerate(node_ids)}

    H = G.copy()
    for u, v, k, data in list(H.edges(keys=True, data=True)):
        if "length" not in data:
            data["length"] = 100.0

    for src in node_ids:
        if src not in G:
            continue
        lengths = nx.single_source_dijkstra_path_length(
            H.to_undirected(), src, weight="length"
        )
        i = idx_of[src]
        for tgt, dist_m in lengths.items():
            if tgt in idx_of:
                out[i, idx_of[tgt]] = dist_m / 1000.0

    np.fill_diagonal(out, 0.0)
    return out

def _block_segments_through_circle(d: np.ndarray, coords: np.ndarray,
                                    center: np.ndarray, radius_km: float) -> None:
    p1 = coords[:, None, :] - center
    p2 = coords[None, :, :] - center
    dvec = p2 - p1
    a = (dvec * dvec).sum(-1)
    b = 2.0 * (p1 * dvec).sum(-1)
    c = (p1 * p1).sum(-1) - radius_km ** 2
    disc = b ** 2 - 4.0 * a * c
    valid = (disc >= 0) & (a > 1e-12)
    disc_safe = np.where(valid, disc, 0.0)
    sd = np.sqrt(disc_safe)
    denom = 2.0 * a + 1e-12
    t1 = np.where(valid, (-b - sd) / denom, 2.0)
    t2 = np.where(valid, (-b + sd) / denom, 2.0)
    crossed = valid & (((t1 >= 0) & (t1 <= 1)) | ((t2 >= 0) & (t2 <= 1)))
    d[crossed] = float("inf")

def load_mississauga_networks(cache_path: str = _CACHE_PATH_DEFAULT,
                              rebuild: bool = False) -> MississaugaNetworks:
    global _NETWORKS_CACHE
    if _NETWORKS_CACHE is not None and not rebuild:
        return _NETWORKS_CACHE

    if os.path.exists(cache_path) and not rebuild:
        try:
            with open(cache_path, "rb") as f:
                nets: MississaugaNetworks = pickle.load(f)
            if not hasattr(nets, "pearson_xy_km"):
                raise AttributeError("stale cache missing pearson_xy_km")
            _NETWORKS_CACHE = nets
            return nets
        except (AttributeError, Exception) as exc:
            print(f"[mississauga] Cache invalid ({exc}), rebuilding from OSM...")

    nets = _build_networks_from_osm(cache_path)
    _NETWORKS_CACHE = nets
    return nets

def create_mississauga_instance(
    n_req: int,
    n_uav: int,
    n_adr: int,
    n_depots_uav: int,
    n_depots_adr: int,
    rng: np.random.Generator,
    networks: Optional[MississaugaNetworks] = None,
    cache_path: str = _CACHE_PATH_DEFAULT,
) -> Dict[str, torch.Tensor]:
    if networks is None:
        networks = load_mississauga_networks(cache_path)

    nets = networks
    n_depots = n_depots_uav + n_depots_adr
    n_total = n_depots + 2 * n_req

    coords_all = nets.coords_km
    rest_pool = np.where(nets.is_restaurant_candidate
                          & (nets.acc_uav_node | nets.acc_adr_node))[0]
    cust_pool = np.where(nets.acc_uav_node | nets.acc_adr_node)[0]
    uav_depot_pool = np.where(nets.acc_uav_node)[0]
    adr_depot_pool = np.where(nets.acc_adr_node)[0]

    if len(rest_pool) < n_req:
        raise RuntimeError(
            f"Mississauga restaurant pool has only {len(rest_pool)} candidates "
            f"but {n_req} pickups requested."
        )
    if len(cust_pool) < n_req:
        raise RuntimeError("Mississauga customer pool too small.")

    pickup_global = rng.choice(rest_pool, size=n_req, replace=False)
    delivery_global = rng.choice(cust_pool, size=n_req, replace=False)
    uav_depot_global = rng.choice(uav_depot_pool, size=n_depots_uav, replace=False)
    adr_depot_global = rng.choice(adr_depot_pool, size=n_depots_adr, replace=False)

    selected_global = np.concatenate([uav_depot_global, adr_depot_global,
                                       pickup_global, delivery_global])

    coords = coords_all[selected_global].astype(np.float32)
    d_uav_km = nets.d_uav_km[np.ix_(selected_global, selected_global)].astype(np.float32)
    d_adr_km = nets.d_adr_km[np.ix_(selected_global, selected_global)].astype(np.float32)

    KM_PER_COORD = SCALE_M_PER_COORD / 1000.0
    d_uav = d_uav_km / KM_PER_COORD
    d_adr = d_adr_km / KM_PER_COORD
    coords_coord = coords / KM_PER_COORD

    acc_uav = nets.acc_uav_node[selected_global].copy()
    acc_adr = nets.acc_adr_node[selected_global].copy()

    demand_vals = rng.uniform(1.0, 6.0, size=n_req).astype(np.float32)
    demand = np.zeros(n_total, dtype=np.float32)
    demand[n_depots:n_depots + n_req] = demand_vals
    demand[n_depots + n_req:] = -demand_vals

    shift_minutes = 120.0
    inter_arrival = rng.exponential(scale=shift_minutes / n_req, size=n_req)
    t_pickup = np.cumsum(inter_arrival).astype(np.float32)
    t_pickup = np.clip(t_pickup, 0, shift_minutes - 10)
    delivery_slack = rng.normal(loc=30.0, scale=5.0, size=n_req).astype(np.float32)
    delivery_slack = np.clip(delivery_slack, 15.0, 60.0)
    t_delivery = (t_pickup + delivery_slack).astype(np.float32)

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
    wind_mag = float(rng.uniform(0, 12.0))
    wind_vec = np.array([wind_mag * math.cos(wind_dir),
                         wind_mag * math.sin(wind_dir)], dtype=np.float32)

    from energy import uav_energy_matrix, adr_energy_matrix
    e_uav = uav_energy_matrix(
        d_uav * SCALE_M_PER_COORD, demand_vals, demand, n_depots, n_req,
        wind_vec, UAV_CRUISE_MPS,
    )
    e_adr = adr_energy_matrix(
        d_adr * SCALE_M_PER_COORD, demand_vals, demand, n_depots, n_req,
        ADR_CRUISE_MPS,
    )

    tp_full = np.zeros(n_total, dtype=np.float32)
    tp_full[n_depots:n_depots + n_req] = t_pickup
    margin_uav = static_signed_margin(tp_full, d_uav, V_UAV_COORD_PER_MIN)
    margin_adr = static_signed_margin(tp_full, d_adr, V_ADR_COORD_PER_MIN)

    high = max(coords_coord.max(), 5.0)
    DIST_SCALE = high * math.sqrt(2)
    ENERGY_SCALE = 200_000.0
    DEMAND_MAX = 6.0
    COORD_SCALE = high
    TIME_SCALE = shift_minutes + 60.0

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

    delivery_slack_node = np.zeros(n_total, dtype=np.float32)
    delivery_slack_node[n_depots:n_depots + n_req] = delivery_slack / shift_minutes
    delivery_slack_node[n_depots + n_req:]         = delivery_slack / shift_minutes

    node_feat = np.concatenate([
        coords_coord / COORD_SCALE,
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

    Q_adr = 10.0
    capacity = np.hstack([np.ones(n_uav, dtype=np.float32) * Q_uav,
                          np.ones(n_adr, dtype=np.float32) * Q_adr])
    BATTERY_UAV = 6500.0
    BATTERY_ADR = 4500.0
    battery = np.hstack([np.ones(n_uav, dtype=np.float32) * BATTERY_UAV,
                         np.ones(n_adr, dtype=np.float32) * BATTERY_ADR])

    inst = {
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
            np.concatenate([np.zeros(n_depots, dtype=np.float32),
                            t_pickup, t_delivery]),
            dtype=torch.float32,
        ),
        'coords_km': torch.tensor(coords, dtype=torch.float32),
        'osm_node_indices': torch.tensor(selected_global, dtype=torch.long),
        'n_depots': n_depots,
        'n_depots_uav': n_depots_uav,
        'n_depots_adr': n_depots_adr,
        'n_req': n_req,
        'is_real_world': True,
    }
    return inst

if __name__ == "__main__":
    nets = load_mississauga_networks()
    print(f"Loaded {len(nets.coords_km)} candidate nodes.")
    print(f"Restaurant candidates: {nets.is_restaurant_candidate.sum()}")
    print(f"UAV-accessible: {nets.acc_uav_node.sum()}")
    print(f"ADR-accessible: {nets.acc_adr_node.sum()}")
    inst = create_mississauga_instance(
        n_req=10, n_uav=2, n_adr=2, n_depots_uav=1, n_depots_adr=1,
        rng=np.random.default_rng(0), networks=nets,
    )
    print(f"Instance x shape: {inst['x'].shape}")
    print(f"edge_attr_uav shape: {inst['edge_attr_uav'].shape}")
    print(f"Map extent (km): {nets.bbox_xy_km}")
