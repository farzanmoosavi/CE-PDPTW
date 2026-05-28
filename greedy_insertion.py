from __future__ import annotations
import math
import copy
from typing import Dict, List, Tuple, Optional, Any

from dispatch_sim import Request, Vehicle, Leg
from reward import request_penalty, operating_cost_per_minute, battery_penalty
from energy import uav_edge_energy, adr_edge_energy
from vrpUpdate import (
    V_UAV_MAX,
    V_ADR_MAX,
    V_UAV_DEPOT,
    V_ADR_DEPOT,
    V_UAV_MIN_PICKUP,
    V_ADR_MIN_PICKUP,
    UAV_LAND_TAKEOFF_MIN,
)

def _get_dist(full_inst: dict, mode: str):
    import torch
    key = 'edge_attr_d' if mode == 'uav' else 'edge_attr_r'
    ea = full_inst.get(key)
    if ea is None:
        return None
    if isinstance(ea, torch.Tensor):
        ea = ea.squeeze(-1).cpu().numpy()
    n = int(round(ea.shape[0] ** 0.5))
    if n * n == ea.shape[0]:
        return ea.reshape(n, n)
    return None

def _wind_vec(full_inst: Dict[str, Any]):
    import numpy as np
    wind = full_inst.get('wind')
    if wind is None:
        return np.zeros(2)
    if hasattr(wind, 'cpu'):
        wind = wind.cpu().numpy()
    wind_mag, wind_dir = float(wind[0]), float(wind[1])
    return np.array([wind_mag * math.cos(wind_dir), wind_mag * math.sin(wind_dir)])

def _service_time(mode: str) -> float:
    return UAV_LAND_TAKEOFF_MIN if mode == 'uav' else 0.0

def _pickup_speed(distance: float, t_now: float, t_target: float, mode: str) -> float:
    if mode == 'uav':
        v_min, v_max = V_UAV_MIN_PICKUP, V_UAV_MAX
    else:
        v_min, v_max = V_ADR_MIN_PICKUP, V_ADR_MAX

    slack = max(t_target - t_now, 0.0)
    if slack <= 0.0 or distance <= 1e-9:
        return v_max
    if distance / max(v_max, 1e-9) >= slack:
        return v_max
    return max(v_min, min(v_max, distance / max(slack, 1e-9)))

def _depot_speed(mode: str) -> float:
    return V_UAV_DEPOT if mode == 'uav' else V_ADR_DEPOT

def _cruise_speed(mode: str) -> float:
    return V_UAV_MAX if mode == 'uav' else V_ADR_MAX

def _battery_threshold(vehicle: Vehicle) -> float:
    base = vehicle.battery_init if vehicle.battery_init is not None else vehicle.battery
    frac = 0.25 if vehicle.mode == 'uav' else 0.20
    return frac * float(base)

def _nearest_feasible_depot(from_node: int, dm, n_depots: int) -> Optional[int]:
    best_node = None
    best_d = float('inf')
    for dep in range(n_depots):
        d = float(dm[from_node][dep])
        if d < best_d and d < 1e9:
            best_d = d
            best_node = dep
    return best_node

def _energy_kj(mode: str, distance: float, payload: float, wind_vec, speed: float) -> float:
    if mode == 'uav':
        return uav_edge_energy(distance, payload, wind_vec, speed) / 1000.0
    return adr_edge_energy(distance, payload) / 1000.0

def _simulate_leg(
    vehicle_state: Vehicle,
    req: Optional[Request],
    leg_type: str,
    to_node: int,
    full_inst: dict,
) -> Tuple[Optional[Leg], Vehicle, bool]:
    dm = _get_dist(full_inst, vehicle_state.mode)
    if dm is None:
        return None, float('inf'), False

    from_node = vehicle_state.current_node
    distance = float(dm[from_node][to_node])
    if distance >= 1e9:
        return None, float('inf'), False

    depart = float(vehicle_state.current_time)
    n_depots = int(full_inst['n_depots'])

    if to_node < n_depots:
        speed = _depot_speed(vehicle_state.mode)
        travel_time = distance / max(speed, 1e-9)
        raw_arrive = depart + travel_time
        complete = raw_arrive
        if from_node >= n_depots:
            complete += 10.0 if vehicle_state.mode == 'uav' else 20.0
    elif leg_type == 'pickup':
        target = req.t_pickup if req is not None else depart
        speed = _pickup_speed(distance, depart, target, vehicle_state.mode)
        travel_time = distance / max(speed, 1e-9)
        raw_arrive = depart + travel_time
        complete = max(raw_arrive, target) + _service_time(vehicle_state.mode)
    else:
        speed = _cruise_speed(vehicle_state.mode)
        travel_time = distance / max(speed, 1e-9)
        raw_arrive = depart + travel_time
        complete = raw_arrive + _service_time(vehicle_state.mode)

    wind = _wind_vec(full_inst)
    payload = 0.0
    if bool(vehicle_state.onboard_requests) and req is not None:
        payload = float(req.demand)
    energy_used = _energy_kj(vehicle_state.mode, distance, payload, wind, speed)

    next_state = copy.deepcopy(vehicle_state)
    next_state.current_node = to_node
    next_state.current_time = complete
    if to_node < n_depots:
        next_state.battery = float(next_state.battery_init if next_state.battery_init is not None else next_state.battery)
    else:
        next_state.battery = max(0.0, float(next_state.battery) - energy_used)

    if leg_type == 'pickup' and req is not None:
        next_state.onboard_requests.add(req.req_id)
    elif leg_type == 'delivery' and req is not None and req.req_id in next_state.onboard_requests:
        next_state.onboard_requests.discard(req.req_id)

    leg = Leg(
        request_id=-1 if req is None else req.req_id,
        vehicle_id=vehicle_state.vehicle_id,
        leg_type=leg_type if to_node >= n_depots else 'depot',
        from_node=from_node,
        to_node=to_node,
        t_depart=depart,
        t_arrive=complete,
        travel_time=travel_time,
        operating_cost=operating_cost_per_minute(vehicle_state.mode == 'uav') * travel_time,
        t_arrive_raw=raw_arrive,
        energy_used=energy_used,
    )
    return leg, next_state, True

def _materialize_ops(
    vehicle: Vehicle,
    ops: List[Tuple[int, str]],
    requests: Dict[int, Request],
    full_inst: dict,
) -> Tuple[List[Leg], float, bool]:
    vehicle_state = copy.deepcopy(vehicle)
    if vehicle_state.battery_init is None:
        vehicle_state.battery_init = float(vehicle_state.battery)

    n_depots = int(full_inst['n_depots'])
    dm = _get_dist(full_inst, vehicle.mode)
    if dm is None:
        return [], float('inf'), False

    legs: List[Leg] = []
    total_cost = 0.0
    pickup_actual: Dict[int, float] = {}

    for req_id, leg_type in ops:
        req = requests[req_id]
        to_node = req.pickup_node if leg_type == 'pickup' else req.delivery_node

        need_depot = False
        if vehicle_state.current_node >= n_depots and to_node >= n_depots:
            threshold = _battery_threshold(vehicle_state)
            distance = float(dm[vehicle_state.current_node][to_node])
            if distance >= 1e9:
                return [], float('inf'), False
            speed = _pickup_speed(distance, vehicle_state.current_time, req.t_pickup, vehicle.mode) if leg_type == 'pickup' else _cruise_speed(vehicle.mode)
            payload = float(req.demand) if bool(vehicle_state.onboard_requests) else 0.0
            projected_energy = _energy_kj(vehicle.mode, distance, payload, _wind_vec(full_inst), speed)
            if vehicle_state.battery <= threshold or (vehicle_state.battery - projected_energy) < threshold:
                need_depot = True

        if need_depot:
            depot = _nearest_feasible_depot(vehicle_state.current_node, dm, n_depots)
            if depot is None:
                return [], float('inf'), False
            depot_leg, vehicle_state, ok = _simulate_leg(vehicle_state, None, 'depot', depot, full_inst)
            if not ok:
                return [], float('inf'), False
            total_cost += depot_leg.operating_cost + battery_penalty(vehicle_state.battery, _battery_threshold(vehicle_state))
            legs.append(depot_leg)

        leg, vehicle_state, ok = _simulate_leg(vehicle_state, req, leg_type, to_node, full_inst)
        if not ok:
            return [], float('inf'), False

        total_cost += leg.operating_cost
        total_cost += battery_penalty(vehicle_state.battery, _battery_threshold(vehicle_state))

        if leg_type == 'pickup':
            pickup_actual[req_id] = leg.t_arrive_raw
        else:
            total_cost += request_penalty(
                T_pickup_actual=pickup_actual.get(req_id, req.t_pickup),
                t_p=req.t_pickup,
                T_delivery_actual=leg.t_arrive,
                t_d=req.t_delivery,
            )

        legs.append(leg)

    return legs, total_cost, True

def _route_ops_from_vehicle(vehicle: Vehicle, active_requests: Dict[int, Request]) -> List[Tuple[int, str]]:
    ops: List[Tuple[int, str]] = []
    for onboard_rid in vehicle.onboard_requests:
        req = active_requests.get(onboard_rid)
        if req is not None and req.status in ('onboard', 'delivery_committed', 'pickup_committed'):
            ops.append((req.req_id, 'delivery'))
    return ops

def _insertion_cost(
    vehicle: Vehicle,
    ops: List[Tuple[int, str]],
    req: Request,
    insert_pos: int,
    active_requests: Dict[int, Request],
    full_inst: dict,
) -> float:
    candidate_ops = list(ops)
    candidate_ops[insert_pos:insert_pos] = [(req.req_id, 'pickup'), (req.req_id, 'delivery')]

    _, candidate_cost, ok = _materialize_ops(vehicle, candidate_ops, active_requests, full_inst)
    if not ok:
        return float('inf')

    _, base_cost, ok_base = _materialize_ops(vehicle, ops, active_requests, full_inst)
    if not ok_base:
        base_cost = 0.0

    return max(0.0, candidate_cost - base_cost)

class GreedyInsertion:

    def solve_ops(self, residual: dict) -> Dict[int, List[Tuple[int, str]]]:
        vehicles = residual['vehicles']
        active_requests = residual['active_requests']

        ops = {vid: _route_ops_from_vehicle(veh, active_requests) for vid, veh in vehicles.items()}

        to_insert = [
            r for r in active_requests.values()
            if r.status == 'waiting_for_pickup'
        ]
        to_insert.sort(key=lambda r: (r.t_delivery, r.t_pickup, r.req_id))

        for req in to_insert:
            best_cost = float('inf')
            best_vid = None
            best_pos = None

            for vid, veh in vehicles.items():
                vehicle_ops = ops[vid]
                for pos in range(len(vehicle_ops) + 1):
                    cost = _insertion_cost(
                        veh, vehicle_ops, req, pos, active_requests, residual['full_instance']
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_vid = vid
                        best_pos = pos

            if best_vid is not None and best_cost < float('inf'):
                ops[best_vid][best_pos:best_pos] = [(req.req_id, 'pickup'), (req.req_id, 'delivery')]

        return ops

    def solve(self, residual: dict) -> Dict[int, List[Leg]]:
        vehicles = residual['vehicles']
        active_requests = residual['active_requests']
        full_inst = residual['full_instance']

        ops = self.solve_ops(residual)
        routes: Dict[int, List[Leg]] = {}
        for vid, veh in vehicles.items():
            legs, _, ok = _materialize_ops(veh, ops.get(vid, []), active_requests, full_inst)
            routes[vid] = legs if ok else []
        return routes
