from __future__ import annotations

import copy
import dataclasses
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from reward import request_penalty, operating_cost_per_minute, battery_penalty, ALPHA_U
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

@dataclass
class Request:
    req_id: int
    t_arrival: float
    t_pickup: float
    t_delivery: float
    pickup_node: int
    delivery_node: int
    demand: float
    status: str = "unrevealed"
    assigned_vehicle: Optional[int] = None
    T_pickup_actual: Optional[float] = None
    T_delivery_actual: Optional[float] = None

    VALID_TRANSITIONS = {
        "unrevealed": {"waiting_for_pickup"},
        "waiting_for_pickup": {"pickup_committed"},
        "pickup_committed": {"onboard"},
        "onboard": {"delivery_committed"},
        "delivery_committed": {"delivered"},
        "delivered": set(),
    }

    def transition(self, new_status: str) -> None:
        if new_status not in self.VALID_TRANSITIONS[self.status]:
            raise ValueError(f"Invalid transition {self.status} -> {new_status} for request {self.req_id}")
        self.status = new_status

@dataclass
class Leg:
    request_id: int
    vehicle_id: int
    leg_type: str
    from_node: int
    to_node: int
    t_depart: float
    t_arrive: float
    travel_time: float = 0.0
    operating_cost: float = 0.0
    t_arrive_raw: float = 0.0
    energy_used: float = 0.0
    load_before: float = 0.0
    load_after: float = 0.0

@dataclass
class Vehicle:
    vehicle_id: int
    mode: str
    current_node: int
    current_time: float
    battery: float
    load: float
    capacity: float
    committed_leg: Optional[Leg] = None
    onboard_request: Optional[int] = None
    battery_init: Optional[float] = None
    allowed_depots: Optional[Tuple[int, ...]] = None
    onboard_requests: Tuple[int, ...] = field(default_factory=tuple)

    def normalized_mode(self) -> str:
        mode = self.mode.lower()
        if mode not in {"uav", "adr"}:
            raise ValueError(f"Unsupported vehicle mode: {self.mode}")
        return mode

    def onboard_set(self) -> set[int]:
        onboard = set(self.onboard_requests)
        if self.onboard_request is not None:
            onboard.add(int(self.onboard_request))
        return onboard

    def set_onboard_set(self, onboard: Iterable[int]) -> None:
        values = tuple(sorted(set(int(x) for x in onboard)))
        self.onboard_requests = values
        self.onboard_request = values[0] if len(values) == 1 else None

@dataclass(frozen=True)
class SyntheticALNSConfig:
    n_uav: int
    n_adr: int
    n_depots_uav: int
    n_depots_adr: int
    depot_sharing: bool = True
    delta_minutes: float = 5.0
    shift_minutes: Optional[float] = None
    alpha_p_ratio: Optional[float] = None

def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)

def _get_distance_matrix(full_instance: Dict[str, Any], mode: str) -> np.ndarray:
    mode = mode.lower()
    key = "edge_attr_d" if mode == "uav" else "edge_attr_r"
    alt_key = "d_uav" if mode == "uav" else "d_adr"

    if key in full_instance:
        raw = _to_numpy(full_instance[key]).squeeze()
        n_total = int(round(math.sqrt(raw.size)))
        if n_total * n_total != raw.size:
            raise ValueError(f"{key} cannot be reshaped into a square matrix.")
        return raw.reshape(n_total, n_total).astype(float)

    if alt_key in full_instance:
        return _to_numpy(full_instance[alt_key]).astype(float)

    raise KeyError(f"No distance matrix found for mode={mode}")

def _get_mask_matrix(full_instance: Dict[str, Any], mode: str) -> Optional[np.ndarray]:
    mode = mode.lower()
    key = "mask_adjacency_uav" if mode == "uav" else "mask_adjacency_adr"

    if key not in full_instance:
        return None

    raw = _to_numpy(full_instance[key]).squeeze()
    n_total = int(round(math.sqrt(raw.size)))
    if n_total * n_total != raw.size:
        return None
    return raw.reshape(n_total, n_total).astype(float)

def _is_arc_feasible(full_instance: Dict[str, Any], mode: str, from_node: int, to_node: int) -> bool:
    dm = _get_distance_matrix(full_instance, mode)
    if from_node < 0 or to_node < 0 or from_node >= dm.shape[0] or to_node >= dm.shape[1]:
        return False

    distance = float(dm[from_node, to_node])
    if not math.isfinite(distance) or distance >= 1e9:
        return False

    mask = _get_mask_matrix(full_instance, mode)
    if mask is not None and mask[from_node, to_node] <= 0.5:
        return False

    return True

def _wind_vec(full_instance: Dict[str, Any]) -> np.ndarray:
    wind = full_instance.get("wind")
    if wind is None:
        return np.zeros(2, dtype=float)

    wind = _to_numpy(wind).reshape(-1).astype(float)
    if len(wind) < 2:
        return np.zeros(2, dtype=float)

    wind_mag, wind_dir = float(wind[0]), float(wind[1])
    return np.array([wind_mag * math.cos(wind_dir), wind_mag * math.sin(wind_dir)], dtype=float)

def _cruise_speed(mode: str) -> float:
    return V_UAV_MAX if mode.lower() == "uav" else V_ADR_MAX

def _depot_speed(mode: str) -> float:
    return V_UAV_DEPOT if mode.lower() == "uav" else V_ADR_DEPOT

def _pickup_speed(distance: float, t_now: float, t_target: float, mode: str) -> float:
    if mode.lower() == "uav":
        v_min, v_max = V_UAV_MIN_PICKUP, V_UAV_MAX
    else:
        v_min, v_max = V_ADR_MIN_PICKUP, V_ADR_MAX

    slack = max(t_target - t_now, 0.0)
    if slack <= 0.0 or distance <= 1e-9:
        return v_max
    if distance / max(v_max, 1e-9) >= slack:
        return v_max
    return max(v_min, min(v_max, distance / max(slack, 1e-9)))

def _service_time(mode: str) -> float:
    return UAV_LAND_TAKEOFF_MIN if mode.lower() == "uav" else 0.0

def _battery_threshold(vehicle: Vehicle) -> float:
    base = vehicle.battery_init if vehicle.battery_init is not None else vehicle.battery
    frac = 0.25 if vehicle.normalized_mode() == "uav" else 0.20
    return frac * float(base)

def _recharge_minutes(vehicle: Vehicle) -> float:
    return 10.0 if vehicle.normalized_mode() == "uav" else 20.0

def _full_battery(vehicle: Vehicle) -> float:
    return float(vehicle.battery_init if vehicle.battery_init is not None else vehicle.battery)

def _edge_energy_kj(
    *,
    mode: str,
    distance: float,
    payload: float,
    wind_vec: np.ndarray,
    speed: float,
) -> float:
    if mode.lower() == "uav":
        return uav_edge_energy(distance, payload, wind_vec, speed) / 1000.0
    return adr_edge_energy(distance, payload) / 1000.0

def _nearest_feasible_depot(from_node: int, vehicle: Vehicle, full_instance: Dict[str, Any]) -> Optional[int]:
    dm = _get_distance_matrix(full_instance, vehicle.normalized_mode())
    n_depots = int(full_instance["n_depots"])

    candidates = vehicle.allowed_depots if vehicle.allowed_depots is not None else tuple(range(n_depots))

    best_node = None
    best_distance = float("inf")

    for depot in candidates:
        if depot < 0 or depot >= n_depots:
            continue
        if not _is_arc_feasible(full_instance, vehicle.normalized_mode(), from_node, depot):
            continue

        distance = float(dm[from_node, depot])
        if distance < best_distance:
            best_distance = distance
            best_node = int(depot)

    return best_node

def _simulate_leg(
    vehicle_state: Vehicle,
    req: Optional[Request],
    leg_type: str,
    to_node: int,
    full_instance: Dict[str, Any],
) -> Tuple[Optional[Leg], Vehicle, bool]:
    mode = vehicle_state.normalized_mode()
    dm = _get_distance_matrix(full_instance, mode)

    from_node = int(vehicle_state.current_node)
    if not _is_arc_feasible(full_instance, mode, from_node, to_node):
        return None, vehicle_state, False

    distance = float(dm[from_node, to_node])
    n_depots = int(full_instance["n_depots"])

    depart = float(vehicle_state.current_time)
    is_depot_leg = to_node < n_depots

    if is_depot_leg:
        speed = _depot_speed(mode)
        travel_time = distance / max(speed, 1e-9)
        raw_arrive = depart + travel_time
        complete = raw_arrive
        if from_node >= n_depots:
            complete += _recharge_minutes(vehicle_state)
    elif leg_type == "pickup":
        target = req.t_pickup if req is not None else depart
        speed = _pickup_speed(distance, depart, target, mode)
        travel_time = distance / max(speed, 1e-9)
        raw_arrive = depart + travel_time
        complete = max(raw_arrive, target) + _service_time(mode)
    else:
        speed = _cruise_speed(mode)
        travel_time = distance / max(speed, 1e-9)
        raw_arrive = depart + travel_time
        complete = raw_arrive + _service_time(mode)

    load_before = float(vehicle_state.load)
    load_after = load_before

    onboard = vehicle_state.onboard_set()

    if leg_type == "pickup" and req is not None:
        if req.req_id in onboard:
            return None, vehicle_state, False
        load_after = load_before + float(req.demand)
        if load_after > vehicle_state.capacity + 1e-9:
            return None, vehicle_state, False
        onboard.add(req.req_id)

    elif leg_type == "delivery" and req is not None:
        if req.req_id not in onboard:
            return None, vehicle_state, False
        load_after = max(0.0, load_before - float(req.demand))
        onboard.remove(req.req_id)

    payload = load_before
    energy_used = _edge_energy_kj(
        mode=mode,
        distance=distance,
        payload=payload,
        wind_vec=_wind_vec(full_instance),
        speed=speed,
    )

    next_state = copy.deepcopy(vehicle_state)
    next_state.current_node = int(to_node)
    next_state.current_time = float(complete)
    next_state.load = float(load_after)
    next_state.set_onboard_set(onboard)

    if is_depot_leg:
        next_state.battery = _full_battery(next_state)
    else:
        next_state.battery = max(0.0, float(next_state.battery) - energy_used)

    leg = Leg(
        request_id=-1 if req is None else int(req.req_id),
        vehicle_id=int(vehicle_state.vehicle_id),
        leg_type="depot" if is_depot_leg else leg_type,
        from_node=from_node,
        to_node=int(to_node),
        t_depart=depart,
        t_arrive=float(complete),
        travel_time=float(travel_time),
        operating_cost=operating_cost_per_minute(mode == "uav") * float(travel_time),
        t_arrive_raw=float(raw_arrive),
        energy_used=float(energy_used),
        load_before=float(load_before),
        load_after=float(load_after),
    )

    return leg, next_state, True

def _needs_depot_before_next_leg(
    vehicle_state: Vehicle,
    req: Request,
    leg_type: str,
    full_instance: Dict[str, Any],
) -> bool:
    n_depots = int(full_instance["n_depots"])
    if vehicle_state.current_node < n_depots:
        return False

    to_node = req.pickup_node if leg_type == "pickup" else req.delivery_node
    mode = vehicle_state.normalized_mode()
    if not _is_arc_feasible(full_instance, mode, vehicle_state.current_node, to_node):
        return False

    dm = _get_distance_matrix(full_instance, mode)
    distance = float(dm[vehicle_state.current_node, to_node])

    if leg_type == "pickup":
        speed = _pickup_speed(distance, vehicle_state.current_time, req.t_pickup, mode)
    else:
        speed = _cruise_speed(mode)

    projected_energy = _edge_energy_kj(
        mode=mode,
        distance=distance,
        payload=float(vehicle_state.load),
        wind_vec=_wind_vec(full_instance),
        speed=speed,
    )

    threshold = _battery_threshold(vehicle_state)
    return (
        float(vehicle_state.battery) <= threshold
        or float(vehicle_state.battery) - projected_energy < threshold
    )

def _materialize_ops(
    vehicle: Vehicle,
    ops: Sequence[Tuple[int, str]],
    requests: Dict[int, Request],
    full_instance: Dict[str, Any],
) -> Tuple[List[Leg], float, bool]:
    vehicle_state = copy.deepcopy(vehicle)
    if vehicle_state.battery_init is None:
        vehicle_state.battery_init = float(vehicle_state.battery)

    legs: List[Leg] = []
    total_cost = 0.0
    pickup_actual: Dict[int, float] = {}

    for req_id, leg_type in ops:
        if req_id not in requests:
            return [], float("inf"), False
        if leg_type not in {"pickup", "delivery"}:
            return [], float("inf"), False

        req = requests[req_id]
        to_node = int(req.pickup_node if leg_type == "pickup" else req.delivery_node)

        if _needs_depot_before_next_leg(vehicle_state, req, leg_type, full_instance):
            depot = _nearest_feasible_depot(vehicle_state.current_node, vehicle_state, full_instance)
            if depot is None:
                return [], float("inf"), False

            depot_leg, vehicle_state, ok = _simulate_leg(
                vehicle_state=vehicle_state,
                req=None,
                leg_type="depot",
                to_node=depot,
                full_instance=full_instance,
            )
            if not ok or depot_leg is None:
                return [], float("inf"), False

            legs.append(depot_leg)
            total_cost += depot_leg.operating_cost
            total_cost += battery_penalty(vehicle_state.battery, _battery_threshold(vehicle_state))

        leg, vehicle_state, ok = _simulate_leg(
            vehicle_state=vehicle_state,
            req=req,
            leg_type=leg_type,
            to_node=to_node,
            full_instance=full_instance,
        )
        if not ok or leg is None:
            return [], float("inf"), False

        legs.append(leg)
        total_cost += leg.operating_cost
        total_cost += battery_penalty(vehicle_state.battery, _battery_threshold(vehicle_state))

        if leg_type == "pickup":
            pickup_actual[req_id] = leg.t_arrive_raw
        else:
            total_cost += request_penalty(
                T_pickup_actual=pickup_actual.get(req_id, req.t_pickup),
                t_p=req.t_pickup,
                T_delivery_actual=leg.t_arrive,
                t_d=req.t_delivery,
            )

    return legs, total_cost, True

def _route_ops_from_vehicle(vehicle: Vehicle, active_requests: Dict[int, Request]) -> List[Tuple[int, str]]:
    ops: List[Tuple[int, str]] = []

    for req_id in sorted(vehicle.onboard_set()):
        req = active_requests.get(req_id)
        if req is not None and req.status in {"onboard", "delivery_committed", "pickup_committed"}:
            ops.append((req.req_id, "delivery"))

    return ops

def _insert_pickup_delivery(
    ops: Sequence[Tuple[int, str]],
    req_id: int,
    pickup_pos: int,
    delivery_pos_after_pickup: int,
) -> List[Tuple[int, str]]:
    candidate = list(ops)
    candidate[pickup_pos:pickup_pos] = [(req_id, "pickup")]
    candidate[delivery_pos_after_pickup:delivery_pos_after_pickup] = [(req_id, "delivery")]
    return candidate

def _all_pickup_delivery_insertions(
    ops: Sequence[Tuple[int, str]],
    req_id: int,
) -> Iterable[List[Tuple[int, str]]]:
    base_len = len(ops)
    for pickup_pos in range(base_len + 1):
        with_pickup = list(ops)
        with_pickup[pickup_pos:pickup_pos] = [(req_id, "pickup")]

        for delivery_pos in range(pickup_pos + 1, len(with_pickup) + 1):
            candidate = list(with_pickup)
            candidate[delivery_pos:delivery_pos] = [(req_id, "delivery")]
            yield candidate

def _best_insertion(
    vehicle: Vehicle,
    ops: Sequence[Tuple[int, str]],
    req: Request,
    active_requests: Dict[int, Request],
    full_instance: Dict[str, Any],
) -> Tuple[float, Optional[List[Tuple[int, str]]]]:
    _, base_cost, ok_base = _materialize_ops(vehicle, ops, active_requests, full_instance)
    if not ok_base:
        base_cost = 0.0

    best_delta = float("inf")
    best_ops: Optional[List[Tuple[int, str]]] = None

    for candidate_ops in _all_pickup_delivery_insertions(ops, req.req_id):
        _, candidate_cost, ok = _materialize_ops(vehicle, candidate_ops, active_requests, full_instance)
        if not ok:
            continue

        delta = max(0.0, candidate_cost - base_cost)
        if delta < best_delta:
            best_delta = delta
            best_ops = candidate_ops

    return best_delta, best_ops

class ALNSSolution:
    def __init__(self, ops: Dict[int, List[Tuple[int, str]]], unassigned: Sequence[int]):
        self.ops = {int(vid): list(route_ops) for vid, route_ops in ops.items()}
        self.unassigned = list(dict.fromkeys(int(rid) for rid in unassigned))
        self._cost_cache: Optional[float] = None

    def copy(self) -> "ALNSSolution":
        return ALNSSolution(self.ops, self.unassigned)

    def invalidate(self) -> None:
        self._cost_cache = None

    def route_cost(
        self,
        vehicles: Dict[int, Vehicle],
        requests: Dict[int, Request],
        full_instance: Dict[str, Any],
    ) -> float:
        if self._cost_cache is not None:
            return self._cost_cache

        total = 0.0
        for vid, vehicle in vehicles.items():
            _, cost, ok = _materialize_ops(vehicle, self.ops.get(vid, []), requests, full_instance)
            if not ok:
                self._cost_cache = float("inf")
                return self._cost_cache
            total += cost

        self._cost_cache = total
        return total

    def objective(
        self,
        vehicles: Dict[int, Vehicle],
        requests: Dict[int, Request],
        full_instance: Dict[str, Any],
    ) -> Tuple[int, float]:
        return len(self.unassigned), self.route_cost(vehicles, requests, full_instance)

def destroy_random(sol: ALNSSolution, n_remove: int, rng: random.Random) -> Tuple[ALNSSolution, List[int]]:
    sol = sol.copy()

    pickup_ids = sorted({
        req_id
        for ops in sol.ops.values()
        for req_id, leg_type in ops
        if leg_type == "pickup"
    })

    n_remove = min(n_remove, len(pickup_ids))
    removed_ids = rng.sample(pickup_ids, n_remove) if n_remove > 0 else []
    removed = set(removed_ids)

    for vid in sol.ops:
        sol.ops[vid] = [op for op in sol.ops[vid] if op[0] not in removed]

    sol.invalidate()
    return sol, removed_ids

def destroy_worst(
    sol: ALNSSolution,
    n_remove: int,
    vehicles: Dict[int, Vehicle],
    requests: Dict[int, Request],
    full_instance: Dict[str, Any],
) -> Tuple[ALNSSolution, List[int]]:
    sol = sol.copy()
    contributions: List[Tuple[float, int]] = []

    for vid, vehicle in vehicles.items():
        base_ops = sol.ops.get(vid, [])
        _, base_cost, ok = _materialize_ops(vehicle, base_ops, requests, full_instance)
        if not ok:
            continue

        route_req_ids = sorted({req_id for req_id, leg_type in base_ops if leg_type == "pickup"})
        for req_id in route_req_ids:
            pruned = [op for op in base_ops if op[0] != req_id]
            _, pruned_cost, ok_pruned = _materialize_ops(vehicle, pruned, requests, full_instance)
            if ok_pruned:
                contributions.append((base_cost - pruned_cost, req_id))

    contributions.sort(reverse=True)
    removed_ids: List[int] = []
    seen: set[int] = set()

    for _, req_id in contributions:
        if req_id not in seen:
            removed_ids.append(req_id)
            seen.add(req_id)
        if len(removed_ids) >= n_remove:
            break

    removed = set(removed_ids)
    for vid in sol.ops:
        sol.ops[vid] = [op for op in sol.ops[vid] if op[0] not in removed]

    sol.invalidate()
    return sol, removed_ids

def destroy_related(
    sol: ALNSSolution,
    n_remove: int,
    requests: Dict[int, Request],
    rng: random.Random,
) -> Tuple[ALNSSolution, List[int]]:
    pickup_ids = sorted({
        req_id
        for ops in sol.ops.values()
        for req_id, leg_type in ops
        if leg_type == "pickup"
    })

    if not pickup_ids:
        return sol.copy(), []

    seed_req_id = rng.choice(pickup_ids)
    seed_req = requests.get(seed_req_id)
    if seed_req is None:
        return sol.copy(), []

    def related_score(req_id: int) -> Tuple[float, float]:
        req = requests[req_id]
        temporal = abs(req.t_pickup - seed_req.t_pickup) + abs(req.t_delivery - seed_req.t_delivery)
        spatial = abs(req.pickup_node - seed_req.pickup_node) + abs(req.delivery_node - seed_req.delivery_node)
        return temporal, spatial

    removed_ids = sorted(pickup_ids, key=related_score)[:n_remove]
    removed = set(removed_ids)

    sol = sol.copy()
    for vid in sol.ops:
        sol.ops[vid] = [op for op in sol.ops[vid] if op[0] not in removed]

    sol.invalidate()
    return sol, removed_ids

def repair_greedy(
    sol: ALNSSolution,
    repair_ids: Sequence[int],
    vehicles: Dict[int, Vehicle],
    requests: Dict[int, Request],
    full_instance: Dict[str, Any],
) -> ALNSSolution:
    sol = sol.copy()
    sol.unassigned = []

    pending = [requests[req_id] for req_id in dict.fromkeys(repair_ids) if req_id in requests]
    pending.sort(key=lambda r: (r.t_delivery, r.t_pickup, r.req_id))

    for req in pending:
        best_delta = float("inf")
        best_vid: Optional[int] = None
        best_ops: Optional[List[Tuple[int, str]]] = None

        for vid, vehicle in vehicles.items():
            delta, candidate_ops = _best_insertion(
                vehicle=vehicle,
                ops=sol.ops.get(vid, []),
                req=req,
                active_requests=requests,
                full_instance=full_instance,
            )
            if candidate_ops is not None and delta < best_delta:
                best_delta = delta
                best_vid = vid
                best_ops = candidate_ops

        if best_vid is not None and best_ops is not None:
            sol.ops[best_vid] = best_ops
        else:
            sol.unassigned.append(req.req_id)

    sol.invalidate()
    return sol

def repair_regret2(
    sol: ALNSSolution,
    repair_ids: Sequence[int],
    vehicles: Dict[int, Vehicle],
    requests: Dict[int, Request],
    full_instance: Dict[str, Any],
) -> ALNSSolution:
    sol = sol.copy()
    sol.unassigned = []

    pending = [requests[req_id] for req_id in dict.fromkeys(repair_ids) if req_id in requests]

    while pending:
        regrets = []

        for req in pending:
            choices = []
            for vid, vehicle in vehicles.items():
                delta, candidate_ops = _best_insertion(
                    vehicle=vehicle,
                    ops=sol.ops.get(vid, []),
                    req=req,
                    active_requests=requests,
                    full_instance=full_instance,
                )
                if candidate_ops is not None:
                    choices.append((delta, vid, candidate_ops))

            choices.sort(key=lambda item: item[0])

            if not choices:
                regrets.append((-float("inf"), req, None, None, float("inf")))
                continue

            c1 = choices[0][0]
            c2 = choices[1][0] if len(choices) > 1 else c1
            regret = c2 - c1
            regrets.append((regret, req, choices[0][1], choices[0][2], c1))

        regrets.sort(key=lambda item: -item[0])
        _, req, best_vid, best_ops, best_delta = regrets[0]
        pending.remove(req)

        if best_vid is not None and best_ops is not None and best_delta < float("inf"):
            sol.ops[int(best_vid)] = best_ops
        else:
            sol.unassigned.append(req.req_id)

    sol.invalidate()
    return sol

class ALNSBaseline:
    def __init__(
        self,
        *,
        seed: int = 42,
        time_budget_small_s: float = 3.0,
        time_budget_large_s: float = 8.0,
        large_threshold: int = 30,
        max_remove: int = 5,
    ):
        self._rng = random.Random(seed)
        self.time_budget_small_s = float(time_budget_small_s)
        self.time_budget_large_s = float(time_budget_large_s)
        self.large_threshold = int(large_threshold)
        self.max_remove = int(max_remove)

    def initial_solution(self, residual: Dict[str, Any]) -> ALNSSolution:
        vehicles: Dict[int, Vehicle] = residual["vehicles"]
        active_requests: Dict[int, Request] = residual["active_requests"]
        full_instance = residual["full_instance"]

        ops = {
            vid: _route_ops_from_vehicle(vehicle, active_requests)
            for vid, vehicle in vehicles.items()
        }

        waiting = [
            req for req in active_requests.values()
            if req.status == "waiting_for_pickup"
        ]
        waiting.sort(key=lambda r: (r.t_delivery, r.t_pickup, r.req_id))

        sol = ALNSSolution(ops, [])

        for req in waiting:
            best_delta = float("inf")
            best_vid = None
            best_ops = None

            for vid, vehicle in vehicles.items():
                delta, candidate_ops = _best_insertion(
                    vehicle=vehicle,
                    ops=sol.ops.get(vid, []),
                    req=req,
                    active_requests=active_requests,
                    full_instance=full_instance,
                )
                if candidate_ops is not None and delta < best_delta:
                    best_delta = delta
                    best_vid = vid
                    best_ops = candidate_ops

            if best_vid is not None and best_ops is not None:
                sol.ops[best_vid] = best_ops
            else:
                sol.unassigned.append(req.req_id)

        sol.invalidate()
        return sol

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        vehicles: Dict[int, Vehicle] = residual["vehicles"]
        active_requests: Dict[int, Request] = residual["active_requests"]
        full_instance = residual["full_instance"]

        n_active = len(active_requests)
        time_budget = self.time_budget_small_s if n_active <= self.large_threshold else self.time_budget_large_s

        current = self.initial_solution(residual)
        best = current.copy()
        best_obj = best.objective(vehicles, active_requests, full_instance)

        destroy_weights = [1.0, 1.0, 1.0]
        repair_weights = [1.0, 1.0]

        t_start = time.perf_counter()
        n_iter = 0

        while time.perf_counter() - t_start < time_budget:
            assigned_pickups = {
                req_id
                for ops in current.ops.values()
                for req_id, leg_type in ops
                if leg_type == "pickup"
            }

            if not assigned_pickups and not current.unassigned:
                break

            n_remove = max(1, min(max(n_active // 4, 1), self.max_remove, len(assigned_pickups) or 1))

            destroy_choice = self._rng.choices(range(3), weights=destroy_weights)[0]
            if destroy_choice == 0:
                candidate, removed_ids = destroy_random(current, n_remove, self._rng)
            elif destroy_choice == 1:
                candidate, removed_ids = destroy_worst(current, n_remove, vehicles, active_requests, full_instance)
            else:
                candidate, removed_ids = destroy_related(current, n_remove, active_requests, self._rng)

            repair_ids = list(dict.fromkeys(list(removed_ids) + list(candidate.unassigned)))
            candidate.unassigned = []

            repair_choice = self._rng.choices(range(2), weights=repair_weights)[0]
            if repair_choice == 0:
                candidate = repair_greedy(candidate, repair_ids, vehicles, active_requests, full_instance)
            else:
                candidate = repair_regret2(candidate, repair_ids, vehicles, active_requests, full_instance)

            cand_obj = candidate.objective(vehicles, active_requests, full_instance)
            curr_obj = current.objective(vehicles, active_requests, full_instance)

            accept = False
            if cand_obj[0] < curr_obj[0]:
                accept = True
            elif cand_obj[0] == curr_obj[0]:
                delta = cand_obj[1] - curr_obj[1]
                temp = max(1.0, curr_obj[1] * 0.01 * math.exp(-n_iter / 100.0))
                accept = delta < 0 or self._rng.random() < math.exp(-delta / temp)

            if accept:
                current = candidate

                if cand_obj < best_obj:
                    best = candidate.copy()
                    best_obj = cand_obj
                    destroy_weights[destroy_choice] = min(destroy_weights[destroy_choice] * 1.1, 5.0)
                    repair_weights[repair_choice] = min(repair_weights[repair_choice] * 1.1, 5.0)

            n_iter += 1

        routes: Dict[int, List[Leg]] = {}
        for vid, vehicle in vehicles.items():
            if vehicle.committed_leg is not None:
                routes[vid] = [vehicle.committed_leg]
                continue

            legs, _, ok = _materialize_ops(vehicle, best.ops.get(vid, []), active_requests, full_instance)
            routes[vid] = legs if ok else []

        return routes

def build_residual_instance(
    requests: Dict[int, Request],
    vehicles: Dict[int, Vehicle],
    current_time: float,
    full_instance: Dict[str, Any],
) -> Dict[str, Any]:
    active_requests = {
        req_id: req
        for req_id, req in requests.items()
        if req.status not in {"unrevealed", "delivered"}
    }

    return {
        "full_instance": full_instance,
        "active_requests": active_requests,
        "vehicles": vehicles,
        "current_time": current_time,
    }

def pick_first_uncommitted_leg(route: Sequence[Leg], requests: Dict[int, Request], n_depots: int) -> Optional[Leg]:
    for leg in route:
        if leg.to_node < n_depots:
            return leg

        req = requests.get(leg.request_id)
        if req is None:
            continue

        if leg.leg_type == "pickup" and req.status == "waiting_for_pickup":
            return leg
        if leg.leg_type == "delivery" and req.status in {"onboard", "delivery_committed"}:
            return leg

    return None

def advance_time(
    t_start: float,
    t_end: float,
    vehicles: Dict[int, Vehicle],
    full_instance: Dict[str, Any],
    requests: Optional[Dict[int, Request]] = None,
) -> Tuple[List[Leg], Dict[int, Vehicle], float, float, float, float]:
    completed_legs: List[Leg] = []
    step_operating_cost = 0.0
    step_battery_penalty = 0.0
    step_operating_cost_uav = 0.0
    step_operating_cost_adr = 0.0
    n_depots = int(full_instance["n_depots"])

    for vehicle in vehicles.values():
        leg = vehicle.committed_leg

        if leg is None:
            vehicle.current_time = max(float(vehicle.current_time), float(t_end))
            continue

        if vehicle.battery_init is None:
            vehicle.battery_init = float(vehicle.battery)

        if leg.t_arrive > t_end + 1e-9:
            vehicle.current_time = max(float(vehicle.current_time), float(t_end))
            continue

        req = requests.get(leg.request_id) if requests is not None else None
        vehicle.current_node = int(leg.to_node)
        vehicle.current_time = float(leg.t_arrive)

        if leg.to_node < n_depots:
            vehicle.battery = _full_battery(vehicle)
        else:
            vehicle.battery = max(0.0, float(vehicle.battery) - float(leg.energy_used))

        onboard = vehicle.onboard_set()

        if req is not None and leg.leg_type == "pickup":
            onboard.add(req.req_id)
            vehicle.load = min(vehicle.capacity, float(vehicle.load) + float(req.demand))
        elif req is not None and leg.leg_type == "delivery":
            onboard.discard(req.req_id)
            vehicle.load = max(0.0, float(vehicle.load) - float(req.demand))

        vehicle.set_onboard_set(onboard)

        leg_cost = float(leg.operating_cost)
        step_operating_cost += leg_cost
        if getattr(vehicle, "mode", None) == "uav":
            step_operating_cost_uav += leg_cost
        else:
            step_operating_cost_adr += leg_cost
        step_battery_penalty += battery_penalty(
            vehicle.battery,
            min_threshold=_battery_threshold(vehicle),
        )
        completed_legs.append(leg)

    return completed_legs, vehicles, step_operating_cost, step_battery_penalty, step_operating_cost_uav, step_operating_cost_adr

def _count_undelivered(requests: Dict[int, Request]) -> int:
    return sum(1 for req in requests.values() if req.status not in {"delivered", "unrevealed"})

def _count_revealed(requests: Dict[int, Request]) -> int:
    return sum(1 for req in requests.values() if req.status != "unrevealed")

class RollingHorizonDispatcher:
    def __init__(
        self,
        solver: ALNSBaseline,
        delta_minutes: float,
        shift_minutes: float = 120.0,
        alpha_p_ratio: Optional[float] = None,
    ):
        self.solver = solver
        self.delta = float(delta_minutes)
        self.T = float(shift_minutes)
        self.alpha_p_ratio = alpha_p_ratio

    def run_shift(
        self,
        arrival_stream: Sequence[Dict[str, Any]],
        fleet_init: Dict[int, Vehicle],
        full_instance: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        t = 0.0
        requests: Dict[int, Request] = {}
        vehicles = {vid: copy.deepcopy(vehicle) for vid, vehicle in fleet_init.items()}

        for vehicle in vehicles.values():
            if vehicle.battery_init is None:
                vehicle.battery_init = float(vehicle.battery)

        episode_log: List[Dict[str, Any]] = []
        total_operating_cost = 0.0
        total_operating_cost_uav = 0.0
        total_operating_cost_adr = 0.0
        total_penalty_cost = 0.0
        total_battery_penalty = 0.0
        _n_uav = sum(1 for v in vehicles.values() if v.mode == "uav")

        while t < self.T:
            for raw in arrival_stream:
                req_id = int(raw["req_id"])
                if raw["t_arrival"] <= t and req_id not in requests:
                    requests[req_id] = Request(
                        req_id=req_id,
                        t_arrival=float(raw["t_arrival"]),
                        t_pickup=float(raw["t_pickup"]),
                        t_delivery=float(raw["t_delivery"]),
                        pickup_node=int(raw["pickup_node"]),
                        delivery_node=int(raw["delivery_node"]),
                        demand=float(raw["demand"]),
                        status="waiting_for_pickup",
                    )

            residual = build_residual_instance(
                requests=requests,
                vehicles=vehicles,
                current_time=t,
                full_instance=full_instance,
            )

            solve_start = time.perf_counter()
            route_plan = self.solver.solve(residual)
            solver_time_s = time.perf_counter() - solve_start

            n_depots = int(full_instance["n_depots"])

            for vid, route in route_plan.items():
                vehicle = vehicles[vid]
                if vehicle.committed_leg is not None:
                    continue

                first_leg = pick_first_uncommitted_leg(route, requests, n_depots)
                if first_leg is None:
                    continue

                vehicle.committed_leg = first_leg
                req = requests.get(first_leg.request_id)

                if req is not None:
                    req.assigned_vehicle = vid
                    if first_leg.leg_type == "pickup" and req.status == "waiting_for_pickup":
                        req.transition("pickup_committed")
                    elif first_leg.leg_type == "delivery" and req.status == "onboard":
                        req.transition("delivery_committed")

            t_next = min(t + self.delta, self.T)
            completed_legs, vehicles, step_operating_cost, step_battery_penalty, step_oc_uav, step_oc_adr = advance_time(
                t_start=t,
                t_end=t_next,
                vehicles=vehicles,
                full_instance=full_instance,
                requests=requests,
            )

            total_operating_cost += step_operating_cost
            for _cl in completed_legs:
                _oc = float(_cl.operating_cost)
                if int(_cl.vehicle_id) < _n_uav:
                    total_operating_cost_uav += _oc
                else:
                    total_operating_cost_adr += _oc
            total_battery_penalty += step_battery_penalty

            completed_payload = []
            step_penalty_cost = 0.0

            for leg in completed_legs:
                req = requests.get(leg.request_id)
                vehicle = vehicles[leg.vehicle_id]

                if req is not None and leg.leg_type == "pickup":
                    req.T_pickup_actual = leg.t_arrive_raw
                    if req.status == "pickup_committed":
                        req.transition("onboard")
                    vehicle.committed_leg = None

                elif req is not None and leg.leg_type == "delivery":
                    req.T_delivery_actual = leg.t_arrive
                    if req.status == "delivery_committed":
                        req.transition("delivered")
                    vehicle.committed_leg = None

                    if req.T_pickup_actual is None:
                        req.T_pickup_actual = req.t_pickup

                    step_penalty_cost += request_penalty(
                        T_pickup_actual=req.T_pickup_actual,
                        t_p=req.t_pickup,
                        T_delivery_actual=req.T_delivery_actual,
                        t_d=req.t_delivery,
                        alpha_p_ratio=self.alpha_p_ratio,
                    )

                else:
                    vehicle.committed_leg = None

                completed_payload.append(dataclasses.asdict(leg))

            total_penalty_cost += step_penalty_cost

            episode_log.append({
                "t": t_next,
                "solver_time_s": solver_time_s,
                "completed": completed_payload,
                "waiting": sum(1 for req in requests.values() if req.status == "waiting_for_pickup"),
                "pickup_committed": sum(1 for req in requests.values() if req.status == "pickup_committed"),
                "onboard": sum(1 for req in requests.values() if req.status == "onboard"),
                "delivery_committed": sum(1 for req in requests.values() if req.status == "delivery_committed"),
                "delivered": sum(1 for req in requests.values() if req.status == "delivered"),
                "operating_cost_step": step_operating_cost,
                "penalty_cost_step": step_penalty_cost,
                "battery_penalty_step": step_battery_penalty,
            })

            t = t_next

        n_undelivered = _count_undelivered(requests)
        undelivered_penalty = ALPHA_U * n_undelivered
        episode_log.append({
            "summary": True,
            "total_revealed": _count_revealed(requests),
            "total_delivered": sum(1 for req in requests.values() if req.status == "delivered"),
            "undelivered": n_undelivered,
            "operating_cost": total_operating_cost,
            "operating_cost_uav": total_operating_cost_uav,
            "operating_cost_adr": total_operating_cost_adr,
            "penalty_cost": total_penalty_cost + total_battery_penalty,
            "battery_penalty_cost": total_battery_penalty,
            "undelivered_penalty": undelivered_penalty,
            "total_cost": total_operating_cost + total_penalty_cost + total_battery_penalty + undelivered_penalty,
            "request_history": [
                {
                    "req_id": req.req_id,
                    "status": req.status,
                    "t_arrival": req.t_arrival,
                    "t_pickup_target": req.t_pickup,
                    "t_delivery_target": req.t_delivery,
                    "T_pickup_actual": req.T_pickup_actual,
                    "T_delivery_actual": req.T_delivery_actual,
                    "assigned_vehicle": req.assigned_vehicle,
                }
                for req in sorted(requests.values(), key=lambda item: item.req_id)
                if req.status != "unrevealed"
            ],
        })

        return episode_log

def build_static_arrival_stream_from_instance(full_instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    n_depots = int(full_instance["n_depots"])
    n_req = int(full_instance["n_req"])

    demand = _to_numpy(full_instance["demand"]).reshape(-1).astype(float)
    time_window = _to_numpy(full_instance["time_window"]).reshape(-1).astype(float)

    stream = []
    for req_id in range(n_req):
        pickup_node = n_depots + req_id
        delivery_node = n_depots + n_req + req_id

        stream.append({
            "req_id": req_id,
            "t_arrival": 0.0,
            "t_pickup": float(time_window[pickup_node]),
            "t_delivery": float(time_window[delivery_node]),
            "pickup_node": pickup_node,
            "delivery_node": delivery_node,
            "demand": abs(float(demand[pickup_node])),
        })

    return stream


def build_dynamic_arrival_stream_from_instance(full_instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Like the static version but uses pre-generated t_arrival per request (staggered arrivals).

    Falls back to t_arrival=0.0 if the instance has no 't_arrival' key (backward compat).
    """
    n_depots = int(full_instance["n_depots"])
    n_req = int(full_instance["n_req"])

    demand = _to_numpy(full_instance["demand"]).reshape(-1).astype(float)
    time_window = _to_numpy(full_instance["time_window"]).reshape(-1).astype(float)

    if "t_arrival" in full_instance:
        t_arrival_arr = _to_numpy(full_instance["t_arrival"]).reshape(-1).astype(float)
    else:
        t_arrival_arr = None

    stream = []
    for req_id in range(n_req):
        pickup_node = n_depots + req_id
        delivery_node = n_depots + n_req + req_id
        t_arr = float(t_arrival_arr[req_id]) if t_arrival_arr is not None else 0.0
        stream.append({
            "req_id": req_id,
            "t_arrival": t_arr,
            "t_pickup": float(time_window[pickup_node]),
            "t_delivery": float(time_window[delivery_node]),
            "pickup_node": pickup_node,
            "delivery_node": delivery_node,
            "demand": abs(float(demand[pickup_node])),
        })

    return sorted(stream, key=lambda x: x["t_arrival"])


def build_initial_fleet_from_instance(
    full_instance: Dict[str, Any],
    config: SyntheticALNSConfig,
) -> Dict[int, Vehicle]:
    n_depots = int(full_instance["n_depots"])
    expected_depots = config.n_depots_uav + config.n_depots_adr
    if expected_depots != n_depots:
        raise ValueError(f"Depot split mismatch: config={expected_depots}, instance={n_depots}")

    capacity = _to_numpy(full_instance["capacity"]).reshape(-1).astype(float)
    battery = _to_numpy(full_instance["battery"]).reshape(-1).astype(float)

    n_agents = config.n_uav + config.n_adr
    if len(capacity) != n_agents or len(battery) != n_agents:
        raise ValueError("Vehicle count mismatch with capacity/battery arrays.")

    all_depots = tuple(range(n_depots))
    uav_depots = tuple(range(config.n_depots_uav))
    adr_depots = tuple(range(config.n_depots_uav, n_depots))

    fleet: Dict[int, Vehicle] = {}

    for i in range(config.n_uav):
        allowed = all_depots if config.depot_sharing else uav_depots
        start_node = allowed[i % len(allowed)]
        fleet[i] = Vehicle(
            vehicle_id=i,
            mode="uav",
            current_node=int(start_node),
            current_time=0.0,
            battery=float(battery[i]),
            load=0.0,
            capacity=float(capacity[i]),
            battery_init=float(battery[i]),
            allowed_depots=allowed,
        )

    for j in range(config.n_adr):
        idx = config.n_uav + j
        allowed = all_depots if config.depot_sharing else adr_depots
        start_node = allowed[j % len(allowed)]
        fleet[idx] = Vehicle(
            vehicle_id=idx,
            mode="adr",
            current_node=int(start_node),
            current_time=0.0,
            battery=float(battery[idx]),
            load=0.0,
            capacity=float(capacity[idx]),
            battery_init=float(battery[idx]),
            allowed_depots=allowed,
        )

    return fleet

def infer_shift_minutes(full_instance: Dict[str, Any], extra_minutes: float = 120.0) -> float:
    time_window = _to_numpy(full_instance["time_window"]).reshape(-1).astype(float)
    return float(np.nanmax(time_window) + extra_minutes)

def solve_static_instance_with_alns(
    full_instance: Dict[str, Any],
    config: SyntheticALNSConfig,
    *,
    solver: Optional[ALNSBaseline] = None,
) -> List[Dict[str, Any]]:
    solver = solver if solver is not None else ALNSBaseline(seed=42)
    arrival_stream = build_dynamic_arrival_stream_from_instance(full_instance)
    fleet = build_initial_fleet_from_instance(full_instance, config)

    shift_minutes = config.shift_minutes
    if shift_minutes is None:
        shift_minutes = infer_shift_minutes(full_instance)

    dispatcher = RollingHorizonDispatcher(
        solver=solver,
        delta_minutes=config.delta_minutes,
        shift_minutes=shift_minutes,
        alpha_p_ratio=config.alpha_p_ratio,
    )

    return dispatcher.run_shift(
        arrival_stream=arrival_stream,
        fleet_init=fleet,
        full_instance=full_instance,
    )

def summarize_episode_log(episode_log: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not episode_log:
        return {}

    summary = dict(episode_log[-1]) if episode_log[-1].get("summary") else {}
    if not summary:
        return {}

    return {
        "total_revealed": summary.get("total_revealed"),
        "total_delivered": summary.get("total_delivered"),
        "undelivered": summary.get("undelivered"),
        "operating_cost": summary.get("operating_cost"),
        "penalty_cost": summary.get("penalty_cost"),
        "battery_penalty_cost": summary.get("battery_penalty_cost"),
        "total_cost": summary.get("total_cost"),
        "request_history": summary.get("request_history"),
    }

def build_toy_full_instance() -> Dict[str, Any]:
    coordinates = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [1.0, 0.0],
        [1.2, 0.1],
        [2.0, 0.0],
        [2.2, 0.1],
    ], dtype=np.float32)

    n_depots = 2
    n_req = 2
    n_total = n_depots + 2 * n_req

    diff = coordinates[:, None, :] - coordinates[None, :, :]
    d_uav = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)
    d_adr = d_uav.copy()

    mask = np.ones((n_total, n_total), dtype=np.float32)
    np.fill_diagonal(mask, 0.0)

    demand = np.zeros(n_total, dtype=np.float32)
    demand[n_depots + 0] = 1.0
    demand[n_depots + 1] = 1.2
    demand[n_depots + n_req + 0] = -1.0
    demand[n_depots + n_req + 1] = -1.2

    time_window = np.zeros(n_total, dtype=np.float32)
    time_window[n_depots + 0] = 4.0
    time_window[n_depots + 1] = 4.5
    time_window[n_depots + n_req + 0] = 18.0
    time_window[n_depots + n_req + 1] = 18.5

    return {
        "edge_attr_d": d_uav.reshape(-1, 1),
        "edge_attr_r": d_adr.reshape(-1, 1),
        "mask_adjacency_uav": mask.reshape(-1, 1),
        "mask_adjacency_adr": mask.reshape(-1, 1),
        "demand": demand.reshape(-1, 1),
        "time_window": time_window.reshape(-1, 1),
        "capacity": np.array([5.0, 10.0], dtype=np.float32),
        "battery": np.array([6500.0, 4500.0], dtype=np.float32),
        "wind": np.array([0.0, 0.0], dtype=np.float32),
        "n_depots": n_depots,
        "n_req": n_req,
    }

def demo_toy_alns() -> None:
    full_instance = build_toy_full_instance()
    config = SyntheticALNSConfig(
        n_uav=1,
        n_adr=1,
        n_depots_uav=1,
        n_depots_adr=1,
        depot_sharing=False,
        delta_minutes=2.0,
        shift_minutes=80.0,
    )

    solver = ALNSBaseline(
        seed=42,
        time_budget_small_s=1.0,
        time_budget_large_s=2.0,
    )

    log = solve_static_instance_with_alns(full_instance, config, solver=solver)
    print(summarize_episode_log(log))

if __name__ == "__main__":
    demo_toy_alns()
