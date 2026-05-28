from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from VRP_Actor import Model
from creat_vrp import creat_data, build_dynamic_sub_batch
from vrpUpdate import PC_uav, PC_adr, V_UAV_MAX, V_ADR_MAX, V_UAV_DEPOT, V_ADR_DEPOT, UAV_LAND_TAKEOFF_MIN

RUNG_ORDER = ['A', 'B', 'C', 'D']

RUNG_CONFIG = {
    'A': dict(n_req=5,  n_uav=2,  n_adr=2, n_depots_uav=1, n_depots_adr=1, batch_size=256),
    'B': dict(n_req=10, n_uav=4,  n_adr=3, n_depots_uav=1, n_depots_adr=1, batch_size=256),
    'C': dict(n_req=25, n_uav=5,  n_adr=4, n_depots_uav=2, n_depots_adr=2, batch_size=128),
    'D': dict(n_req=60, n_uav=10, n_adr=8, n_depots_uav=3, n_depots_adr=3, batch_size=64),
}

EDGE_DIM_NAMES = ['margin', 'distance', 'energy', 'temporal_gap']

V_UAV  = 6.0
V_ADR  = 2.49
ALPHA_1 = 0.60
ALPHA_2 = 0.10
ALPHA_E = 0.02
ALPHA_P = 0.10
ALPHA_D = 0.15

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    return torch.device('cpu')

def reward_breakdown(
    time_window, tour_indices, edge_attr_d, edge_attr_r, time_tensor, n_uav,
    num_depots: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    dev = tour_indices.device
    batch_size, n_agent, steps = tour_indices.size()
    n_adr = n_agent - n_uav

    if num_depots is None:
        num_depots = int((time_window == 0).sum().item() // batch_size)
    num_nodes = time_window.size(1) - num_depots
    num_pickup = num_nodes // 2

    tw     = time_window.view(batch_size, num_depots + num_nodes)
    ea_d   = edge_attr_d.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)
    ea_r   = edge_attr_r.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)

    prev_idx = tour_indices[:, :, :-1]
    cur_idx  = tour_indices[:, :, 1:]

    is_depot     = cur_idx < num_depots
    is_pickup    = (cur_idx >= num_depots) & (cur_idx < num_depots + num_pickup)
    is_delivery  = cur_idx >= num_depots + num_pickup
    is_customer  = is_pickup | is_delivery
    is_consec_depot = is_depot & (prev_idx < num_depots)

    batch_u = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, n_uav).unsqueeze(2)
    batch_a = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, n_adr).unsqueeze(2)
    dis_u = ea_d[batch_u, prev_idx[:, :n_uav],  cur_idx[:, :n_uav]]
    dis_a = ea_r[batch_a, prev_idx[:, n_uav:],  cur_idx[:, n_uav:]]

    prev_time = torch.cat(
        [torch.zeros(batch_size, n_agent, 1, device=dev), time_tensor[:, :, :-1]], dim=2
    )
    tw_cur = torch.gather(tw.unsqueeze(1).expand(-1, n_agent, -1), 2, cur_idx)

    tt_u = dis_u / V_UAV
    tt_a = dis_a / V_ADR

    if n_uav > 0:
        pick_u = is_pickup[:, :n_uav]
        if pick_u.any():
            slack_u = (tw_cur[:, :n_uav] - prev_time[:, :n_uav]).clamp(min=0)
            speed_u = (dis_u / slack_u.clamp(min=1e-8)).clamp(min=2.4, max=V_UAV)
            rush_u  = (slack_u <= 0) | (dis_u / V_UAV >= slack_u)
            speed_u = torch.where(rush_u, torch.full_like(speed_u, V_UAV), speed_u)
            tt_u = tt_u.clone()
            tt_u[pick_u] = (dis_u / speed_u)[pick_u]

    if n_adr > 0:
        pick_a = is_pickup[:, n_uav:]
        if pick_a.any():
            slack_a = (tw_cur[:, n_uav:] - prev_time[:, n_uav:]).clamp(min=0)
            speed_a = (dis_a / slack_a.clamp(min=1e-8)).clamp(min=0.6, max=V_ADR)
            rush_a  = (slack_a <= 0) | (dis_a / V_ADR >= slack_a)
            speed_a = torch.where(rush_a, torch.full_like(speed_a, V_ADR), speed_a)
            tt_a = tt_a.clone()
            tt_a[pick_a] = (dis_a / speed_a)[pick_a]

    tt = torch.cat([tt_u, tt_a], dim=1)
    tt = torch.where(is_consec_depot, torch.zeros_like(tt), tt)

    alpha_op = torch.cat([
        torch.full((batch_size, n_uav, steps - 1), ALPHA_1, device=dev),
        torch.full((batch_size, n_adr,  steps - 1), ALPHA_2, device=dev),
    ], dim=1)
    op_cost_per_agent = (alpha_op * tt).sum(dim=2)

    arrival = prev_time + tt
    completion = arrival.clone()
    if n_uav > 0:
        completion[:, :n_uav] = completion[:, :n_uav] + UAV_LAND_TAKEOFF_MIN * is_customer[:, :n_uav].float()

    early_p = F.relu(tw_cur - arrival) * is_pickup.float()
    late_p  = F.relu(arrival - tw_cur) * is_pickup.float()
    late_d  = F.relu(completion - tw_cur) * is_delivery.float()

    for t in (early_p, late_p, late_d):
        t[is_consec_depot.expand_as(t)] = 0.0

    op_uav = op_cost_per_agent[:, :n_uav].sum(dim=1)
    op_adr = op_cost_per_agent[:, n_uav:].sum(dim=1)
    pk_late = (ALPHA_P * late_p.sum(dim=2)).sum(dim=1)
    dl_late = (ALPHA_D * late_d.sum(dim=2)).sum(dim=1)
    pk_early = (ALPHA_E * early_p.sum(dim=2)).sum(dim=1)
    total = op_uav + op_adr + pk_late + dl_late + pk_early

    n_late_pickup   = (late_p.sum(dim=2)  > 0).sum(dim=1).float()
    n_late_delivery = (late_d.sum(dim=2)  > 0).sum(dim=1).float()

    return {
        'operating_uav':    op_uav,
        'operating_adr':    op_adr,
        'pickup_late_cost': pk_late,
        'delivery_late_cost': dl_late,
        'pickup_early_cost': pk_early,
        'total':            total,
        'n_late_pickup_agents':    n_late_pickup,
        'n_late_delivery_agents':  n_late_delivery,
    }

BATTERY_UAV_KJ   = 6500.0
BATTERY_ADR_KJ   = 4500.0
BATTERY_MIN_UAV  = 0.25
BATTERY_MIN_ADR  = 0.20
SCALE_M_PER_COORD = 200.0

def battery_violation_stats(
    tour_idx: torch.Tensor,
    edge_attr_d: torch.Tensor,
    edge_attr_r: torch.Tensor,
    demand: torch.Tensor,
    n_uav: int,
    n_depots: int,
) -> Dict[str, float]:
    dev = tour_idx.device
    B, n_agents, steps = tour_idx.shape
    n_adr = n_agents - n_uav

    n_total_sq = edge_attr_d.numel() // B
    n_total = int(round(n_total_sq ** 0.5))
    ea_d = edge_attr_d.view(B, n_total, n_total)
    ea_r = edge_attr_r.view(B, n_total, n_total)

    demand_flat = demand.view(B, n_total)

    batt_init = torch.tensor(
        [BATTERY_UAV_KJ] * n_uav + [BATTERY_ADR_KJ] * n_adr,
        dtype=torch.float32, device=dev
    )
    battery = batt_init.unsqueeze(0).expand(B, -1).clone()
    threshold = torch.tensor(
        [BATTERY_UAV_KJ * BATTERY_MIN_UAV] * n_uav
        + [BATTERY_ADR_KJ * BATTERY_MIN_ADR] * n_adr,
        dtype=torch.float32, device=dev
    )

    any_violation = torch.zeros(B, n_agents, dtype=torch.bool, device=dev)

    b_idx = torch.arange(B, device=dev)

    for s in range(1, steps):
        prev = tour_idx[:, :, s - 1]
        curr = tour_idx[:, :, s]

        is_depot = curr < n_depots

        dis_u = ea_d[b_idx.unsqueeze(1).expand(B, n_uav),
                     prev[:, :n_uav], curr[:, :n_uav]]
        dis_a = ea_r[b_idx.unsqueeze(1).expand(B, n_adr),
                     prev[:, n_uav:], curr[:, n_uav:]]

        dis_u_m = dis_u * SCALE_M_PER_COORD
        dis_a_m = dis_a * SCALE_M_PER_COORD

        v_u_ms = torch.full_like(dis_u_m, 20.0)
        v_a_ms = torch.full_like(dis_a_m, 8.3)
        v_u_ms = torch.where(is_depot[:, :n_uav], v_u_ms * 0.5, v_u_ms)
        v_a_ms = torch.where(is_depot[:, n_uav:], v_a_ms * 0.5, v_a_ms)

        tt_u = dis_u_m / v_u_ms.clamp(min=1e-9) / 60.0
        tt_a = dis_a_m / v_a_ms.clamp(min=1e-9) / 60.0

        payload_u = demand_flat[b_idx.unsqueeze(1).expand(B, n_uav), curr[:, :n_uav]].abs()
        payload_a = demand_flat[b_idx.unsqueeze(1).expand(B, n_adr), curr[:, n_uav:]].abs()

        power_u = PC_uav(payload_u, v_u_ms)
        power_a = PC_adr(payload_a, v_a_ms)

        dep_u = tt_u * power_u * 60.0
        dep_a = tt_a * power_a * 60.0

        dep = torch.cat([dep_u, dep_a], dim=1)
        battery = battery - dep

        any_violation |= (battery < threshold.unsqueeze(0))

    violation_rate  = any_violation.float().mean().item()
    uav_viol_rate   = any_violation[:, :n_uav].float().mean().item() if n_uav > 0 else 0.0
    adr_viol_rate   = any_violation[:, n_uav:].float().mean().item() if n_adr > 0 else 0.0

    final_frac_u = (battery[:, :n_uav] / batt_init[:n_uav]).mean().item() if n_uav > 0 else 1.0
    final_frac_a = (battery[:, n_uav:] / batt_init[n_uav:]).mean().item() if n_adr > 0 else 1.0

    return {
        'battery_violation_rate':     violation_rate,
        'battery_violation_uav':      uav_viol_rate,
        'battery_violation_adr':      adr_viol_rate,
        'battery_remaining_frac_uav': final_frac_u,
        'battery_remaining_frac_adr': final_frac_a,
    }

def mode_assignment_stats(
    actions: torch.Tensor,
    demand:  torch.Tensor,
    n_uav: int,
    n_depots: int,
    n_req: int,
    coords: Optional[torch.Tensor] = None,
    n_depots_uav: int = 1,
) -> Dict[str, float]:
    B, n_agents, steps = actions.shape
    n_pickup = n_req

    uav_visit  = torch.zeros(B, n_pickup, dtype=torch.bool, device=actions.device)
    adr_visit  = torch.zeros(B, n_pickup, dtype=torch.bool, device=actions.device)

    for ag in range(n_agents):
        ag_actions = actions[:, ag, :]
        is_uav = ag < n_uav
        for s in range(steps):
            node = ag_actions[:, s]
            is_pk = (node >= n_depots) & (node < n_depots + n_pickup)
            req_idx = (node - n_depots).clamp(0, n_pickup - 1)
            if is_uav:
                uav_visit[is_pk, req_idx[is_pk]] = True
            else:
                adr_visit[is_pk, req_idx[is_pk]] = True

    assigned_uav = uav_visit & ~adr_visit
    assigned_adr = adr_visit & ~uav_visit
    shared       = uav_visit & adr_visit
    visited      = uav_visit | adr_visit

    total_visited = visited.float().sum().item()
    if total_visited == 0:
        return {}

    pk_demand = demand[:, n_depots:n_depots + n_pickup]

    flat_demand = pk_demand[visited].cpu().numpy()
    flat_uav    = assigned_uav[visited].cpu().numpy().astype(float)

    stats: Dict[str, float] = {
        'uav_frac_overall': float(assigned_uav.sum().item() / total_visited),
        'adr_frac_overall': float(assigned_adr.sum().item() / total_visited),
        'shared_frac':      float(shared.sum().item() / total_visited),
    }

    if len(flat_demand) > 0:
        quintiles = np.percentile(flat_demand, [0, 20, 40, 60, 80, 100])
        for q in range(5):
            lo, hi = quintiles[q], quintiles[q + 1]
            mask = (flat_demand >= lo) & (flat_demand <= hi)
            if mask.sum() > 0:
                stats[f'uav_frac_demand_Q{q+1}'] = float(flat_uav[mask].mean())

    if coords is not None:
        coords_np = coords.cpu().numpy()
        depot_xy = coords_np[:, :n_depots_uav, :]
        pickup_xy = coords_np[:, n_depots:n_depots + n_pickup, :]
        diff = pickup_xy[:, :, None, :] - depot_xy[:, None, :, :]
        dist_to_depot = np.sqrt((diff ** 2).sum(-1)).min(axis=-1)
        flat_dist = dist_to_depot[visited.cpu().numpy()]
        if len(flat_dist) > 0:
            quartiles = np.percentile(flat_dist, [0, 25, 50, 75, 100])
            for q in range(4):
                lo, hi = quartiles[q], quartiles[q + 1]
                mask = (flat_dist >= lo) & (flat_dist <= hi)
                if mask.sum() > 0:
                    stats[f'uav_frac_dist_Q{q+1}'] = float(flat_uav[mask].mean())

    return stats

def ablate_batch(batch: Dict, zero_dims: List[int]) -> Dict:
    ablated = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    for key in ('edge_attr_uav', 'edge_attr_adr'):
        if key in ablated:
            for d in zero_dims:
                ablated[key][..., d] = 0.0
    return ablated

@torch.no_grad()
def evaluate_rung(
    model: Model,
    rung: str,
    n_test: int,
    device: torch.device,
    batch_size: int = 256,
    seed: int = 9999,
    edge_zero_dims: Optional[List[int]] = None,
    compute_mode_stats: bool = False,
    dynamic: bool = False,
) -> Dict[str, float]:
    cfg = RUNG_CONFIG[rung]
    n_req, n_uav, n_adr = cfg['n_req'], cfg['n_uav'], cfg['n_adr']
    n_depots_uav, n_depots_adr = cfg['n_depots_uav'], cfg['n_depots_adr']
    n_depots = n_depots_uav + n_depots_adr

    loader = creat_data(
        n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
        n_test, batch_size=batch_size, streaming=False,
        use_cache=False, seed=seed, shuffle=False,
    )

    model.eval()
    all_metrics: Dict[str, List[float]] = defaultdict(list)
    mode_stats_accum: Dict[str, List[float]] = defaultdict(list)
    n_vis_per_instance: List[int] = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        if edge_zero_dims:
            batch = ablate_batch(batch, edge_zero_dims)

        n_vis = n_req
        if dynamic:
            n_vis = int(torch.randint(max(1, n_req // 4), n_req + 1, (1,)).item())
            batch = build_dynamic_sub_batch(batch, n_vis, n_req, n_depots)

        B = batch['x'].shape[0]
        n_vis_per_instance.extend([n_vis] * B)

        tour_idx, _, time_tensor = model(batch, n_uav, n_adr, greedy=True)
        comps = reward_breakdown(
            batch['time_window'], tour_idx.detach(),
            batch['edge_attr_d'], batch['edge_attr_r'],
            time_tensor, n_uav, num_depots=n_depots,
        )

        for k, v in comps.items():
            all_metrics[k].extend(v.cpu().numpy().tolist())

        batt_stats = battery_violation_stats(
            tour_idx.detach(),
            batch['edge_attr_d'], batch['edge_attr_r'],
            batch['demand'], n_uav, n_depots,
        )
        for k, v in batt_stats.items():
            all_metrics[k].append(v)

        if compute_mode_stats:
            ms = mode_assignment_stats(
                tour_idx.detach(), batch['demand'],
                n_uav, n_depots, n_vis,  # n_vis: correct count for sub-batch
                coords=batch['x'][:, :, :2],
                n_depots_uav=n_depots_uav,
            )
            for k, v in ms.items():
                mode_stats_accum[k].append(v)

    result: Dict[str, float] = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        result[f'{k}_mean'] = float(arr.mean())
        result[f'{k}_std']  = float(arr.std())

    _cost_bases = ('total', 'operating_uav', 'operating_adr',
                   'pickup_late_cost', 'delivery_late_cost')
    if dynamic and n_vis_per_instance:
        # Each instance may have a different n_vis; normalise per visible request.
        n_vis_arr = np.array(n_vis_per_instance, dtype=float)
        result['dynamic'] = True
        result['n_vis_mean'] = float(n_vis_arr.mean())
        for base in _cost_bases:
            raw = np.array(all_metrics.get(base, []))
            if len(raw) > 0:
                per_vis = raw / n_vis_arr[:len(raw)]
                result[f'{base}_mean_per_req'] = float(per_vis.mean())
                result[f'{base}_std_per_req']  = float(per_vis.std())
    else:
        for sfx in ('mean', 'std'):
            for base in _cost_bases:
                key = f'{base}_{sfx}'
                if key in result:
                    result[f'{key}_per_req'] = result[key] / n_req

    n_agents = n_uav + n_adr
    result['pct_late_pickup_agents']   = result.get('n_late_pickup_agents_mean', 0) / n_agents
    result['pct_late_delivery_agents'] = result.get('n_late_delivery_agents_mean', 0) / n_agents

    for k, vals in mode_stats_accum.items():
        result[f'mode_{k}'] = float(np.mean(vals))

    result['rung'] = rung
    result['n_req'] = n_req
    return result

@torch.no_grad()
def evaluate_rung_sampling(
    model: Model,
    rung: str,
    n_test: int,
    device: torch.device,
    n_samples: int,
    batch_size: int = 32,
    chunk: int = 128,
    seed: int = 9999,
    edge_zero_dims: Optional[List[int]] = None,
) -> Dict[str, float]:
    cfg = RUNG_CONFIG[rung]
    n_req, n_uav, n_adr = cfg['n_req'], cfg['n_uav'], cfg['n_adr']
    n_depots_uav, n_depots_adr = cfg['n_depots_uav'], cfg['n_depots_adr']
    n_depots = n_depots_uav + n_depots_adr

    loader = creat_data(
        n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
        n_test, batch_size=batch_size, streaming=False,
        use_cache=False, seed=seed, shuffle=False,
    )

    model.eval()
    all_costs: List[float] = []

    n_full, remainder = divmod(n_samples, chunk)
    chunk_sizes = [chunk] * n_full + ([remainder] if remainder > 0 else [])

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        if edge_zero_dims:
            batch = ablate_batch(batch, edge_zero_dims)

        B = batch['x'].size(0)

        x_uav, x_adr = model.encoder(batch)

        num_nodes  = x_uav.shape[1] - n_depots

        dec_keys = ['capacity', 'demand', 'battery', 'time_window',
                    'edge_attr_d', 'edge_attr_r', 'x']

        best_cost: Optional[torch.Tensor] = None

        for c in chunk_sizes:
            xu = x_uav.repeat_interleave(c, dim=0)
            xa = x_adr.repeat_interleave(c, dim=0)

            exp = {k: batch[k].repeat_interleave(c, dim=0) for k in dec_keys}

            acc_uav_exp = exp['x'][:, :, 8]
            acc_adr_exp = exp['x'][:, :, 9]

            actions, _, time_tensor = model.decoder(
                xu, xa,
                exp['capacity'], exp['demand'], exp['battery'],
                exp['time_window'], n_depots, num_nodes,
                n_uav, n_adr,
                exp['edge_attr_d'], exp['edge_attr_r'],
                T=1.0, greedy=False,
                acc_uav=acc_uav_exp, acc_adr=acc_adr_exp,
                parallel_select=True,
                n_depots_uav=n_depots_uav,
            )

            comps = reward_breakdown(
                exp['time_window'], actions.detach(),
                exp['edge_attr_d'], exp['edge_attr_r'],
                time_tensor, n_uav, num_depots=n_depots,
            )
            cost_chunk = comps['total'].cpu().view(B, c).min(dim=1).values

            if best_cost is None:
                best_cost = cost_chunk
            else:
                best_cost = torch.minimum(best_cost, cost_chunk)

        all_costs.extend((best_cost / n_req).tolist())

    costs_arr = np.array(all_costs)
    return {
        'total_mean_per_req': float(costs_arr.mean()),
        'total_std_per_req':  float(costs_arr.std()),
        'n_samples':          n_samples,
        'chunk':              chunk,
        'rung':               rung,
        'n_req':              n_req,
    }

def _build_model(device: torch.device, state_dict=None, arch: str = 'hetgat') -> Model:
    edge_dim = 1 if arch == 'simplegat' else 4
    model = Model(
        input_node_dim=11,
        hidden_node_dim=128,
        input_edge_dim=edge_dim,
        hidden_edge_dim=16,
        conv_layers=3,
        arch=arch,
    ).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model

def _load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    if any(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    return sd, ckpt.get('epoch', '?'), ckpt.get('costs', [])

def _print_mode_specialisation(res: Dict) -> None:
    """Print demand-quartile and distance-quartile UAV-fraction table from mode stats."""
    demand_keys = [f'mode_uav_frac_demand_Q{q}' for q in range(1, 6)]
    dist_keys   = [f'mode_uav_frac_dist_Q{q}'   for q in range(1, 5)]

    has_demand = any(k in res for k in demand_keys)
    has_dist   = any(k in res for k in dist_keys)

    if has_demand:
        print('  Mode specialisation by demand quintile (UAV frac ↓ = heavy→ADR):')
        row = '    '
        for i, k in enumerate(demand_keys, 1):
            row += f'Q{i}={res.get(k, float("nan")):.3f}  '
        print(row)

    if has_dist:
        print('  Mode specialisation by distance quartile (UAV frac ↑ = far→UAV):')
        row = '    '
        for i, k in enumerate(dist_keys, 1):
            row += f'Q{i}={res.get(k, float("nan")):.3f}  '
        print(row)

    if has_demand or has_dist:
        # Summarise direction: expected UAV_Q1_demand < UAV_Q5_demand (lighter→UAV)
        if has_demand:
            q1_d = res.get('mode_uav_frac_demand_Q1', float('nan'))
            q5_d = res.get('mode_uav_frac_demand_Q5', float('nan'))
            direction = 'CORRECT (light→UAV)' if q1_d > q5_d else 'REVERSED — check training'
            print(f'  Demand specialisation direction: Q1={q1_d:.3f} vs Q5={q5_d:.3f} → {direction}')
        if has_dist:
            q1_r = res.get('mode_uav_frac_dist_Q1', float('nan'))
            q4_r = res.get('mode_uav_frac_dist_Q4', float('nan'))
            direction = 'CORRECT (far→UAV)' if q4_r > q1_r else 'REVERSED — check training'
            print(f'  Distance specialisation direction: Q1={q1_r:.3f} vs Q4={q4_r:.3f} → {direction}')


def _save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'Saved: {path}')

def main() -> None:
    parser = argparse.ArgumentParser(description='CE-PDPTW HetGAT evaluation')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint.pth')
    parser.add_argument('--rung', default='A', choices=RUNG_ORDER, help='Training rung of the checkpoint')
    parser.add_argument('--n-test', type=int, default=2048, help='Test instances per rung')
    parser.add_argument('--generalize', action='store_true',
                        help='Test on larger problem scales (same fleet, larger n_req)')
    parser.add_argument('--ablate-edges', action='store_true',
                        help='Run edge-feature ablation study')
    parser.add_argument('--mode-analysis', action='store_true',
                        help='Track UAV vs ADR mode assignment')
    parser.add_argument('--out-dir', default='eval_results', help='Output directory')
    parser.add_argument('--arch', default='hetgat', choices=['hetgat', 'simplegat'],
                        help='Model architecture of the checkpoint (hetgat or simplegat)')
    parser.add_argument('--n-samples', type=int, default=1,
                        help='Sampling-decode repeats per instance (1=greedy only, '
                             '128/1280 adds best-of-K sampling result)')
    parser.add_argument('--sample-chunk', type=int, default=128,
                        help='Instances per forward pass during sampling '
                             '(higher=faster but more GPU memory; default 128)')
    parser.add_argument('--dynamic', action='store_true',
                        help='Evaluate on partial-visibility sub-batches '
                             '(n_vis ~ U[n_req/4, n_req]) to simulate on-demand dispatch; '
                             'costs reported per visible request. Pairs with --mode-analysis '
                             'to verify specialisation holds under partial information.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = _get_device()
    print(f'Device: {device}')

    sd, epoch, costs = _load_checkpoint(args.checkpoint, device)
    print(f'Loaded checkpoint — rung={args.rung}, epoch={epoch}')
    if costs:
        print(f'  Training val costs (last 5): {[f"{c:.4f}" for c in costs[-5:]]}')

    model = _build_model(device, sd, arch=args.arch)
    model.eval()

    rows_main: List[Dict] = []
    rows_ablate: List[Dict] = []
    rows_generalize: List[Dict] = []

    _eval_label = 'dynamic-dispatch' if args.dynamic else 'static-decode'
    _n_req_rung = RUNG_CONFIG[args.rung]['n_req']
    print(f'\n=== In-distribution: Rung {args.rung}  [{_eval_label}] ===')
    if args.dynamic:
        print(f'  Dynamic mode: n_vis ~ U[{max(1, _n_req_rung // 4)}, {_n_req_rung}]')
    t0 = time.time()
    res = evaluate_rung(
        model, args.rung, args.n_test, device,
        compute_mode_stats=args.mode_analysis,
        dynamic=args.dynamic,
    )
    res['config'] = _eval_label
    res['elapsed_s'] = time.time() - t0
    rows_main.append(res)

    _dyn_tag = ' (per vis req)' if args.dynamic else ''
    print(f'  Total cost/req{_dyn_tag}:   {res["total_mean_per_req"]:.4f} ± {res["total_std_per_req"]:.4f}')
    print(f'  Operating UAV/req{_dyn_tag}: {res["operating_uav_mean_per_req"]:.4f}')
    print(f'  Operating ADR/req{_dyn_tag}: {res["operating_adr_mean_per_req"]:.4f}')
    print(f'  Pickup late cost/req{_dyn_tag}: {res["pickup_late_cost_mean_per_req"]:.4f}')
    print(f'  Delivery late cost/req{_dyn_tag}: {res["delivery_late_cost_mean_per_req"]:.4f}')
    print(f'  % agents late pickup:   {res["pct_late_pickup_agents"]:.3f}')
    print(f'  % agents late delivery: {res["pct_late_delivery_agents"]:.3f}')
    print(f'  Battery violation rate: {res.get("battery_violation_rate_mean", 0):.3f}  '
          f'(UAV={res.get("battery_violation_uav_mean", 0):.3f}, '
          f'ADR={res.get("battery_violation_adr_mean", 0):.3f})')
    print(f'  Battery remaining frac: UAV={res.get("battery_remaining_frac_uav_mean", 1):.3f}, '
          f'ADR={res.get("battery_remaining_frac_adr_mean", 1):.3f}')
    if args.dynamic:
        print(f'  Mean visible requests per instance: {res.get("n_vis_mean", 0):.1f}')
    if args.mode_analysis and 'mode_uav_frac_overall' in res:
        print(f'  UAV assignment frac (overall): {res["mode_uav_frac_overall"]:.3f}')
        _print_mode_specialisation(res)

    rows_sampling: List[Dict] = []
    if args.n_samples > 1:
        print(f'\n=== Sampling decode: Rung {args.rung}, K={args.n_samples} ===')
        t0 = time.time()
        sampling_batch = max(8, min(32, RUNG_CONFIG[args.rung]['batch_size'] // 4))
        res_s = evaluate_rung_sampling(
            model, args.rung, args.n_test, device,
            n_samples=args.n_samples,
            batch_size=sampling_batch,
            chunk=args.sample_chunk,
        )
        res_s['elapsed_s'] = time.time() - t0
        res_s['config'] = f'sampling_k{args.n_samples}'
        rows_sampling.append(res_s)

        greedy_cost  = res['total_mean_per_req']
        sampling_cost = res_s['total_mean_per_req']
        gap_pct = 100.0 * (greedy_cost - sampling_cost) / max(greedy_cost, 1e-9)
        print(f'  Greedy cost/req:   {greedy_cost:.4f}')
        print(f'  Sampling cost/req: {sampling_cost:.4f} ± {res_s["total_std_per_req"]:.4f}')
        print(f'  Improvement over greedy: {gap_pct:.2f}%')
        print(f'  Elapsed: {res_s["elapsed_s"]:.1f}s')

    if args.generalize:
        print('\n=== Generalisation study ===')
        train_cfg = RUNG_CONFIG[args.rung]
        gen_sizes = [10, 20, 35, 50]
        for n_req_gen in gen_sizes:
            if n_req_gen == train_cfg['n_req']:
                continue
            gen_cfg = {
                'n_req':        n_req_gen,
                'n_uav':        train_cfg['n_uav'],
                'n_adr':        train_cfg['n_adr'],
                'n_depots_uav': train_cfg['n_depots_uav'],
                'n_depots_adr': train_cfg['n_depots_adr'],
                'batch_size':   max(32, 256 // max(n_req_gen // 5, 1)),
            }
            print(f'  n_req={n_req_gen}, fleet={gen_cfg["n_uav"]}U+{gen_cfg["n_adr"]}A ...', end=' ', flush=True)

            loader = creat_data(
                gen_cfg['n_req'], gen_cfg['n_uav'], gen_cfg['n_adr'],
                gen_cfg['n_depots_uav'], gen_cfg['n_depots_adr'],
                args.n_test,
                batch_size=gen_cfg['batch_size'],
                streaming=False, use_cache=False, seed=8888, shuffle=False,
            )

            costs_gen = []
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with torch.no_grad():
                    tour_idx, _, time_tensor = model(batch, gen_cfg['n_uav'], gen_cfg['n_adr'], greedy=True)
                    comps = reward_breakdown(
                        batch['time_window'], tour_idx.detach(),
                        batch['edge_attr_d'], batch['edge_attr_r'],
                        time_tensor, gen_cfg['n_uav'],
                        num_depots=gen_cfg['n_depots_uav'] + gen_cfg['n_depots_adr'],
                    )
                costs_gen.extend((comps['total'] / n_req_gen).cpu().numpy().tolist())

            mean_cost = float(np.mean(costs_gen))
            std_cost  = float(np.std(costs_gen))
            print(f'cost/req={mean_cost:.4f} ± {std_cost:.4f}')
            rows_generalize.append({
                'train_rung':       args.rung,
                'n_req_test':       n_req_gen,
                'n_uav':            gen_cfg['n_uav'],
                'n_adr':            gen_cfg['n_adr'],
                'total_per_req_mean': mean_cost,
                'total_per_req_std':  std_cost,
            })

        for rung_gen in RUNG_ORDER:
            if rung_gen == args.rung:
                continue
            print(f'  Full rung {rung_gen} ...', end=' ', flush=True)
            try:
                res_gen = evaluate_rung(
                    model, rung_gen, min(args.n_test, 512), device,
                    batch_size=RUNG_CONFIG[rung_gen]['batch_size'],
                )
                print(f'cost/req={res_gen["total_mean_per_req"]:.4f} ± {res_gen["total_std_per_req"]:.4f}')
                rows_generalize.append({
                    'train_rung':         args.rung,
                    'n_req_test':         RUNG_CONFIG[rung_gen]['n_req'],
                    'n_uav':              RUNG_CONFIG[rung_gen]['n_uav'],
                    'n_adr':              RUNG_CONFIG[rung_gen]['n_adr'],
                    'total_per_req_mean': res_gen['total_mean_per_req'],
                    'total_per_req_std':  res_gen['total_std_per_req'],
                    'rung_test':          rung_gen,
                })
            except Exception as e:
                print(f'SKIPPED ({e})')

    if args.ablate_edges:
        print('\n=== Edge-embedding ablation ===')
        ablation_configs = [
            ('full',              []),
            ('no_margin',         [0]),
            ('no_distance',       [1]),
            ('no_energy',         [2]),
            ('no_temporal_gap',   [3]),
            ('no_margin+energy',  [0, 2]),
            ('no_distance+energy',[1, 2]),
            ('distance_only',     [0, 2]),
        ]
        for name, zero_dims in ablation_configs:
            print(f'  {name:25s} ...', end=' ', flush=True)
            res_abl = evaluate_rung(
                model, args.rung, args.n_test, device,
                edge_zero_dims=zero_dims if zero_dims else None,
                dynamic=args.dynamic,
            )
            cost = res_abl['total_mean_per_req']
            std  = res_abl['total_std_per_req']
            print(f'cost/req={cost:.4f} ± {std:.4f}')
            rows_ablate.append({
                'config':               name,
                'zeroed_dims':          str(zero_dims),
                'total_per_req_mean':   cost,
                'total_per_req_std':    std,
                'operating_uav_per_req': res_abl['operating_uav_mean_per_req'],
                'operating_adr_per_req': res_abl['operating_adr_mean_per_req'],
                'pickup_late_per_req':   res_abl['pickup_late_cost_mean_per_req'],
                'delivery_late_per_req': res_abl['delivery_late_cost_mean_per_req'],
                'pct_late_pickup':       res_abl['pct_late_pickup_agents'],
                'pct_late_delivery':     res_abl['pct_late_delivery_agents'],
                'battery_violation_rate': res_abl.get('battery_violation_rate_mean', 0),
                'battery_remaining_uav':  res_abl.get('battery_remaining_frac_uav_mean', 1),
                'battery_remaining_adr':  res_abl.get('battery_remaining_frac_adr_mean', 1),
            })

    prefix = os.path.join(args.out_dir, f'rung{args.rung}')
    _save_csv(rows_main,       f'{prefix}_baseline.csv')
    if rows_generalize:
        _save_csv(rows_generalize, f'{prefix}_generalization.csv')
    if rows_ablate:
        _save_csv(rows_ablate,     f'{prefix}_edge_ablation.csv')
    if rows_sampling:
        _save_csv(rows_sampling,   f'{prefix}_sampling_k{args.n_samples}.csv')

    print('\n=== Summary ===')
    print(f'{"Metric":<40} {"Value":>12}')
    print('-' * 54)
    for k, v in rows_main[0].items():
        if isinstance(v, float):
            print(f'{k:<40} {v:>12.5f}')

    if rows_sampling:
        print(f'\n=== Sampling K={args.n_samples} summary ===')
        print(f'{"total_mean_per_req":<40} {rows_sampling[0]["total_mean_per_req"]:>12.5f}')
        print(f'{"total_std_per_req":<40} {rows_sampling[0]["total_std_per_req"]:>12.5f}')
        gap = 100.0 * (rows_main[0]["total_mean_per_req"] - rows_sampling[0]["total_mean_per_req"]) \
              / max(rows_main[0]["total_mean_per_req"], 1e-9)
        print(f'{"improvement_over_greedy_%":<40} {gap:>12.2f}')

    _print_rl_paper_table(
        rung=args.rung,
        n_test=args.n_test,
        greedy_res=rows_main[0],
        sampling_res=rows_sampling[0] if rows_sampling else None,
        out_dir=args.out_dir,
    )

def _print_rl_paper_table(
    rung: str,
    n_test: int,
    greedy_res: Dict,
    sampling_res: Optional[Dict],
    out_dir: str,
) -> None:
    cfg = RUNG_CONFIG[rung]
    n_req = cfg['n_req']
    n_uav = cfg['n_uav']
    n_adr = cfg['n_adr']

    COL = dict(method=22, n=6, service=9, cost_req=10, tw_pu=9, tw_dl=9, uav_pct=7, time=12)

    header = (
        f"{'Method':<{COL['method']}} "
        f"{'N':>{COL['n']}} "
        f"{'Service%':>{COL['service']}} "
        f"{'Cost/req':>{COL['cost_req']}} "
        f"{'TW-PU%':>{COL['tw_pu']}} "
        f"{'TW-DL%':>{COL['tw_dl']}} "
        f"{'UAV%':>{COL['uav_pct']}} "
        f"{'Time(ms/inst)':>{COL['time']}}"
    )
    sep = '-' * len(header)

    print('\n' + '=' * len(header))
    print(f'  HetGAT-RL paper-format rows  |  Rung {rung}'
          f'  n_req={n_req}  n_uav={n_uav}  n_adr={n_adr}  N={n_test}')
    print('  Append these rows to the baseline table for the full comparison.')
    print('  Service%=100 means all requests have generated routes (offline mode).')
    print('=' * len(header))
    print(header)
    print(sep)

    latex_rows = []

    def _make_row(label: str, res: Dict, elapsed_s: float) -> str:
        cost_req   = float(res.get('total_mean_per_req', 0))
        cost_std   = float(res.get('total_std_per_req', 0))
        tw_pu  = (1.0 - float(res.get('pct_late_pickup_agents', 0))) * 100.0
        tw_dl  = (1.0 - float(res.get('pct_late_delivery_agents', 0))) * 100.0
        uav_op = float(res.get('operating_uav_mean_per_req', 0))
        adr_op = float(res.get('operating_adr_mean_per_req', 0))
        uav_pct = (uav_op / max(uav_op + adr_op, 1e-9)) * 100.0
        time_ms = (elapsed_s / max(n_test, 1)) * 1000.0

        line = (
            f'{label:<{COL["method"]}} '
            f'{n_test:>{COL["n"]}} '
            f'{"100.0":>{COL["service"]}} '
            f'{cost_req:>{COL["cost_req"]}.4f} '
            f'{tw_pu:>{COL["tw_pu"]}.1f} '
            f'{tw_dl:>{COL["tw_dl"]}.1f} '
            f'{uav_pct:>{COL["uav_pct"]}.1f} '
            f'{time_ms:>{COL["time"]}.2f}'
        )
        print(line)

        latex_rows.append(
            f'  {label} & {n_test} & 100.0\\% & {cost_req:.4f}$\\pm${cost_std:.4f}'
            f' & {tw_pu:.1f}\\% & {tw_dl:.1f}\\% & {uav_pct:.1f}\\%'
            f' & {time_ms:.2f}ms \\\\'
        )
        return line

    _make_row('HetGAT-RL (greedy)', greedy_res,
              float(greedy_res.get('elapsed_s', 0)))
    if sampling_res is not None:
        n_k = int(sampling_res.get('n_samples', 0))
        _make_row(f'HetGAT-RL (K={n_k})', sampling_res,
                  float(sampling_res.get('elapsed_s', 0)))

    print(sep)

    os.makedirs(out_dir, exist_ok=True)
    tex_path = os.path.join(out_dir, f'rung{rung}_paper_table_rl.tex')
    latex = [
        r'% HetGAT-RL rows — append to baseline table (paper_table.tex)',
        r'% Columns: Method & N & Service\% & Cost/req & TW-PU\% & TW-DL\% & UAV\% & Time \\',
        r'\midrule',
    ] + latex_rows
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex))
    print(f'  LaTeX snippet saved: {tex_path}')

if __name__ == '__main__':
    main()
