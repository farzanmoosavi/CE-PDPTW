from __future__ import annotations
from typing import Dict, List, Tuple

from dispatch_sim import Request, Vehicle, Leg
from greedy_insertion import (
    _get_dist, _materialize_ops, _route_ops_from_vehicle,
)

def _vehicle_can_serve_node(vehicle: Vehicle, node: int, full_inst: dict) -> bool:
    import torch
    x = full_inst.get('x')
    if x is None:
        return True
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if node >= x.shape[0]:
        return True
    acc_uav = float(x[node, 8])
    acc_adr = float(x[node, 9])
    if vehicle.mode == 'uav':
        return acc_uav > 0.5
    return acc_adr > 0.5

def _distance(vehicle: Vehicle, from_node: int, to_node: int, full_inst: dict) -> float:
    dm = _get_dist(full_inst, vehicle.mode)
    if dm is None:
        return float('inf')
    d = float(dm[from_node][to_node])
    return float('inf') if d >= 1e9 else d

class FIFOSolver:

    def solve_ops(self, residual: dict) -> Dict[int, List[Tuple[int, str]]]:
        vehicles = residual['vehicles']
        active_requests = residual['active_requests']
        full_inst = residual['full_instance']

        ops: Dict[int, List[Tuple[int, str]]] = {
            vid: _route_ops_from_vehicle(veh, active_requests)
            for vid, veh in vehicles.items()
        }

        waiting = [
            r for r in active_requests.values()
            if r.status == 'waiting_for_pickup'
        ]
        waiting.sort(key=lambda r: (r.t_pickup, r.req_id))

        sim_end_node: Dict[int, int] = {}
        for vid, veh in vehicles.items():
            end_node = int(veh.current_node)
            for rid, leg_type in ops[vid]:
                if leg_type == 'delivery':
                    req = active_requests.get(rid)
                    if req is not None:
                        end_node = int(req.delivery_node)
                else:
                    req = active_requests.get(rid)
                    if req is not None:
                        end_node = int(req.pickup_node)
            sim_end_node[vid] = end_node

        sim_load: Dict[int, float] = {vid: float(veh.load) for vid, veh in vehicles.items()}

        for req in waiting:
            best_vid = None
            best_d = float('inf')
            for vid, veh in vehicles.items():
                if not _vehicle_can_serve_node(veh, req.pickup_node, full_inst):
                    continue
                if not _vehicle_can_serve_node(veh, req.delivery_node, full_inst):
                    continue
                if sim_load[vid] + float(req.demand) > float(veh.capacity) + 1e-9:
                    continue
                d = _distance(veh, sim_end_node[vid], int(req.pickup_node), full_inst)
                if d < best_d:
                    best_d = d
                    best_vid = vid

            if best_vid is None:
                continue

            ops[best_vid].extend([(req.req_id, 'pickup'), (req.req_id, 'delivery')])
            sim_end_node[best_vid] = int(req.delivery_node)

        return ops

    def solve(self, residual: dict) -> Dict[int, List[Leg]]:
        vehicles = residual['vehicles']
        active_requests = residual['active_requests']
        full_inst = residual['full_instance']

        ops = self.solve_ops(residual)
        routes: Dict[int, List[Leg]] = {}
        for vid, veh in vehicles.items():
            legs, _cost, ok = _materialize_ops(veh, ops.get(vid, []), active_requests, full_inst)
            routes[vid] = legs if ok else []
        return routes
