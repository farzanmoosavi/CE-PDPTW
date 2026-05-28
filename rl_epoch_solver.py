from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from VRP_Actor import Model
from ce_cpdptw_alns import (
    Leg,
    Request,
    Vehicle,
    _simulate_leg,
)


def _to_tensor(v) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    return torch.tensor(np.asarray(v))


def _build_rl_sub_instance(
    full_instance: Dict[str, Any],
    visible_req_ids: List[int],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    """
    Build a batch=1 tensor dict compatible with Model.forward() containing
    only the requests in visible_req_ids.

    Node layout in sub-instance:
      [0 .. n_depots-1]               depots
      [n_depots .. n_depots+n_vis-1]  pickups  (in sorted_vis order)
      [n_depots+n_vis .. n_sub-1]     deliveries

    Returns (sub_batch, sorted_vis) where sorted_vis[i] is the original
    request id mapped to pickup-slot i in the sub-instance.
    """
    sorted_vis = sorted(visible_req_ids)
    n_vis = len(sorted_vis)
    n_depots = int(full_instance['n_depots'])
    n_req = int(full_instance['n_req'])
    n_total = n_depots + n_req * 2
    n_sub = n_depots + n_vis * 2

    keep = (
        list(range(n_depots)) +
        [n_depots + r for r in sorted_vis] +
        [n_depots + n_req + r for r in sorted_vis]
    )
    keep_t = torch.tensor(keep, dtype=torch.long)

    def _slice_edge(key):
        ea = _to_tensor(full_instance[key])
        D = ea.shape[-1]
        mat = ea.view(n_total, n_total, D)
        return mat[keep_t][:, keep_t].reshape(n_sub * n_sub, D).unsqueeze(0).to(device)

    idx = torch.arange(n_sub)
    ii, jj = torch.meshgrid(idx, idx, indexing='ij')
    ei = torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=0).unsqueeze(0).to(device)

    sub_batch = {
        'x':                  _to_tensor(full_instance['x'])[keep_t].unsqueeze(0).to(device),
        'demand':             _to_tensor(full_instance['demand'])[keep_t].unsqueeze(0).to(device),
        'time_window':        _to_tensor(full_instance['time_window'])[keep_t].unsqueeze(0).to(device),
        'capacity':           _to_tensor(full_instance['capacity']).unsqueeze(0).to(device),
        'battery':            _to_tensor(full_instance['battery']).unsqueeze(0).to(device),
        'wind':               _to_tensor(full_instance.get('wind', torch.zeros(2))).unsqueeze(0).to(device),
        'n_depots':           n_depots,
        'n_depots_uav':       int(full_instance.get('n_depots_uav', n_depots)),
        'n_req':              n_vis,
        'edge_attr_uav':      _slice_edge('edge_attr_uav'),
        'edge_attr_adr':      _slice_edge('edge_attr_adr'),
        'edge_attr_d':        _slice_edge('edge_attr_d'),
        'edge_attr_r':        _slice_edge('edge_attr_r'),
        'mask_adjacency_uav': _slice_edge('mask_adjacency_uav'),
        'mask_adjacency_adr': _slice_edge('mask_adjacency_adr'),
        'edge_index':         ei,
    }
    return sub_batch, sorted_vis


class RLEpochSolver:
    """
    Wraps a trained HetGAT-RL model as a drop-in solver for
    ce_cpdptw_alns.RollingHorizonDispatcher.

    At each epoch the dispatcher calls .solve(residual).  This solver:
      1. Collects requests with status 'waiting_for_pickup'.
      2. Builds a sub-instance tensor (batch=1) with only those requests.
      3. Runs the model greedy → node sequence per agent.
      4. Extracts the first pickup target for each free agent and materialises
         a Leg from the vehicle's actual current position.

    Committed vehicles are returned with an empty route list; the dispatcher
    skips them automatically.
    """

    def __init__(
        self,
        model_path: str,
        n_uav: int,
        n_adr: int,
        n_depots_uav: int,
        n_depots_adr: int,
        input_node_dim: int = 11,
        hidden_node_dim: int = 128,
        input_edge_dim: int = 4,
        hidden_edge_dim: int = 16,
        conv_layers: int = 3,
        arch: str = 'hetgat',
        device: Optional[torch.device] = None,
    ):
        self.n_uav = n_uav
        self.n_adr = n_adr
        self.n_depots_uav = n_depots_uav
        self.n_depots_adr = n_depots_adr
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Model(
            input_node_dim=input_node_dim,
            hidden_node_dim=hidden_node_dim,
            input_edge_dim=input_edge_dim,
            hidden_edge_dim=hidden_edge_dim,
            conv_layers=conv_layers,
            arch=arch,
        ).to(self.device)

        raw = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(raw, dict) and 'model_state_dict' in raw:
            raw = raw['model_state_dict']
        self.model.load_state_dict(raw)
        self.model.eval()
        self._assignment_log: List[Dict[str, Any]] = []

    @torch.no_grad()
    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        vehicles: Dict[int, Vehicle] = residual['vehicles']
        active_requests: Dict[int, Request] = residual['active_requests']
        full_instance = residual['full_instance']

        routes: Dict[int, List[Leg]] = {vid: [] for vid in vehicles}

        waiting = [
            req for req in active_requests.values()
            if req.status == 'waiting_for_pickup'
        ]
        if not waiting:
            return routes

        sub_batch, sorted_vis = _build_rl_sub_instance(
            full_instance, [r.req_id for r in waiting], self.device
        )
        n_vis = len(sorted_vis)
        n_depots = int(full_instance['n_depots'])

        tour_idx, _, _ = self.model(sub_batch, self.n_uav, self.n_adr, greedy=True)
        tour = tour_idx[0].cpu().tolist()

        def _decode_sub_node(sub_node: int):
            if sub_node < n_depots:
                return None, 'depot'
            elif sub_node < n_depots + n_vis:
                return sorted_vis[sub_node - n_depots], 'pickup'
            else:
                return sorted_vis[sub_node - n_depots - n_vis], 'delivery'

        assigned: set = set()
        free_vids = [
            vid for vid in sorted(vehicles.keys())
            if vehicles[vid].committed_leg is None
        ]

        for agent_idx, vid in enumerate(free_vids):
            if agent_idx >= len(tour):
                break
            vehicle = vehicles[vid]
            agent_tour = tour[agent_idx]

            for sub_node in agent_tour[1:]:
                req_id, leg_type = _decode_sub_node(sub_node)
                if req_id is None or leg_type != 'pickup':
                    continue
                if req_id in assigned:
                    continue
                req = active_requests.get(req_id)
                if req is None or req.status != 'waiting_for_pickup':
                    continue

                pickup_node = n_depots + req_id
                leg, _, ok = _simulate_leg(vehicle, req, 'pickup', pickup_node, full_instance)
                if ok and leg is not None:
                    routes[vid] = [leg]
                    assigned.add(req_id)
                    # Track mode assignment for specialisation analysis.
                    n_req_full = int(full_instance['n_req'])
                    n_depots_full = int(full_instance['n_depots'])
                    pickup_node_full = n_depots_full + req_id
                    x_coords = _to_tensor(full_instance['x'])[:, :2]
                    pickup_coord = x_coords[pickup_node_full]
                    depot_coords = x_coords[:n_depots_full]
                    dist_to_depot = float(
                        torch.norm(depot_coords - pickup_coord.unsqueeze(0), dim=1).min()
                    )
                    demand_val = abs(float(
                        _to_tensor(full_instance['demand'])[pickup_node_full]
                    ))
                    self._assignment_log.append({
                        'vehicle_type': 'uav' if vid < self.n_uav else 'adr',
                        'demand': demand_val,
                        'dist_to_depot': dist_to_depot,
                        'req_id': req_id,
                    })
                    break

        return routes

    def reset_log(self) -> None:
        self._assignment_log.clear()

    def get_mode_stats(self) -> Dict[str, Any]:
        """Aggregate mode-specialisation stats over all dispatches since last reset_log()."""
        if not self._assignment_log:
            return {}
        demands = np.array([e['demand']       for e in self._assignment_log], dtype=float)
        dists   = np.array([e['dist_to_depot'] for e in self._assignment_log], dtype=float)
        is_uav  = np.array([e['vehicle_type'] == 'uav' for e in self._assignment_log], dtype=float)

        stats: Dict[str, Any] = {
            'uav_frac_overall': float(is_uav.mean()),
            'adr_frac_overall': float(1.0 - is_uav.mean()),
            'n_assignments':    len(self._assignment_log),
        }

        if len(demands) >= 4:
            for label, arr in [('demand', demands), ('dist_to_depot', dists)]:
                n_q = 5 if label == 'demand' else 4
                breakpoints = np.percentile(arr, np.linspace(0, 100, n_q + 1))
                for q in range(n_q):
                    lo, hi = breakpoints[q], breakpoints[q + 1]
                    mask = (arr >= lo) & (arr <= hi)
                    if mask.sum() > 0:
                        stats[f'uav_frac_{label}_Q{q+1}'] = float(is_uav[mask].mean())

        return stats


def load_rl_epoch_solver(
    model_path: str,
    rung: str,
    arch: str = 'hetgat',
    device: Optional[torch.device] = None,
) -> 'RLEpochSolver':
    """
    Convenience loader that reads fleet config from the standard RUNG_CONFIG.

    Example:
        solver = load_rl_epoch_solver('CE-PDPTW-B-HetGAT/best_actor.pt', rung='B')
        dispatcher = RollingHorizonDispatcher(solver, delta_minutes=5, shift_minutes=120)
        log = dispatcher.run_shift(arrival_stream, fleet, full_instance)
    """
    RUNG_CONFIG = {
        'A': dict(n_uav=2,  n_adr=2, n_depots_uav=1, n_depots_adr=1),
        'B': dict(n_uav=4,  n_adr=3, n_depots_uav=1, n_depots_adr=1),
        'C': dict(n_uav=5,  n_adr=4, n_depots_uav=2, n_depots_adr=2),
        'D': dict(n_uav=10, n_adr=8, n_depots_uav=3, n_depots_adr=3),
    }
    cfg = RUNG_CONFIG[rung]
    edge_dim = 1 if arch == 'simplegat' else 4
    return RLEpochSolver(
        model_path=model_path,
        n_uav=cfg['n_uav'],
        n_adr=cfg['n_adr'],
        n_depots_uav=cfg['n_depots_uav'],
        n_depots_adr=cfg['n_depots_adr'],
        input_edge_dim=edge_dim,
        arch=arch,
        device=device,
    )
