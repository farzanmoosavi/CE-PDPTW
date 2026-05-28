from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any, Literal, Mapping, Sequence
import math
import re

import numpy as np
import torch

import gurobipy as gp
from gurobipy import GRB

Mode = Literal["UAV", "ADR"]
Market = Literal["UAV", "ADR", "shared"]
DepotRole = Literal["start", "mid", "end"]

SCALE_M_PER_COORD = 200.0

UAV_CRUISE_MPS = 20.0
ADR_CRUISE_MPS = 8.3

V_UAV_MAX = UAV_CRUISE_MPS * 60.0 / SCALE_M_PER_COORD
V_ADR_MAX = ADR_CRUISE_MPS * 60.0 / SCALE_M_PER_COORD

V_UAV_DEPOT = V_UAV_MAX / 2.0
V_ADR_DEPOT = V_ADR_MAX / 2.0

V_UAV_MIN_PICKUP = 8.0 * 60.0 / SCALE_M_PER_COORD
V_ADR_MIN_PICKUP = 2.0 * 60.0 / SCALE_M_PER_COORD

UAV_LAND_TAKEOFF_MIN = 2.0
ADR_CUSTOMER_SERVICE_MIN = 0.0

UAV_RECHARGE_MIN = 10.0
ADR_RECHARGE_MIN = 20.0

UAV_LOW_BATTERY_FRAC = 0.25
ADR_LOW_BATTERY_FRAC = 0.20

@dataclass(frozen=True)
class Request:
    id: str
    pickup: str
    delivery: str
    demand: float
    pickup_target: float
    pickup_latest: float
    delivery_target: float
    delivery_latest: float
    market: Market = "shared"
    owner_mode: Mode | None = None

@dataclass(frozen=True)
class Vehicle:
    id: str
    mode: Mode
    capacity: float
    battery_max: float
    battery_min: float
    depot_visit_limit: int
    speed_levels: tuple[str, ...] = ()

@dataclass(frozen=True)
class DepotCopyMeta:
    copy_id: str
    vehicle_id: str
    physical_depot_id: str
    role: DepotRole
    visit_index: int | None

@dataclass(frozen=True)
class CECPDPTWData:
    requests: Sequence[Request]
    vehicles: Sequence[Vehicle]
    physical_depots: Sequence[str]
    uav_depots: Sequence[str]
    adr_depots: Sequence[str]

    travel_time: Mapping[tuple[str, str, str], float]
    energy: Mapping[tuple[str, str, str], float]

    pickup_time_by_speed: Mapping[tuple[str, str, str, str], float] = field(default_factory=dict)
    pickup_energy_by_speed: Mapping[tuple[str, str, str, str], float] = field(default_factory=dict)

    depot_sharing: bool = True

    alpha_uav: float = 0.60
    alpha_adr: float = 0.10
    alpha_early_pickup: float = 0.02
    alpha_late_pickup: float = 0.10
    alpha_late_delivery: float = 0.15
    lambda_battery: float = 0.0

    epsilon_shared: float = 1.0
    epsilon_to_adr: float = 1.0
    epsilon_to_uav: float = 1.0

    uav_customer_service_time: float = UAV_LAND_TAKEOFF_MIN
    adr_customer_service_time: float = ADR_CUSTOMER_SERVICE_MIN
    uav_recharge_duration: float = UAV_RECHARGE_MIN
    adr_recharge_duration: float = ADR_RECHARGE_MIN

    big_m_time: float = 100_000.0
    big_m_load: float = 100_000.0
    big_m_battery: float = 1_000_000.0

@dataclass
class BuiltCECPDPTWModel:
    model: gp.Model
    data: CECPDPTWData
    nodes_by_vehicle: dict[str, list[str]]
    depot_meta: dict[str, DepotCopyMeta]
    depot_copies_by_vehicle: dict[str, list[str]]
    arcs: list[tuple[str, str, str]]
    adaptive_arcs: dict[tuple[str, str, str], tuple[str, ...]]
    x: gp.tupledict
    z: gp.tupledict | None
    use_vehicle: gp.tupledict
    depot_visit: gp.tupledict
    arrival: gp.tupledict
    completion: gp.tupledict
    waiting: gp.tupledict
    load: gp.tupledict
    battery: gp.tupledict
    early_pickup: gp.tupledict
    late_pickup: gp.tupledict
    late_delivery: gp.tupledict
    battery_slack: gp.tupledict
    y_uav: gp.tupledict
    y_adr: gp.tupledict

@dataclass(frozen=True)
class SyntheticAdapterConfig:
    n_uav: int
    n_adr: int
    n_depots_uav: int
    n_depots_adr: int

    depot_visit_limit: int | None = None

    alpha_uav: float = 0.60
    alpha_adr: float = 0.10
    alpha_early_pickup: float = 0.02
    alpha_late_pickup: float = 0.10
    alpha_late_delivery: float = 0.15
    lambda_battery: float = 0.0

    latest_pickup_slack: float = 0.0
    latest_delivery_slack: float = 0.0

    depot_sharing: bool = True
    epsilon_shared: float = 1.0
    epsilon_to_adr: float = 1.0
    epsilon_to_uav: float = 1.0

    use_discrete_pickup_speed: bool = True
    pickup_speed_grid_size: int = 5

    require_destination_access: bool = True
    strict_pair_access: bool = True

    depot_return_energy_payload_policy: Literal["zero", "max_capacity"] = "zero"

    big_m_time: float = 100_000.0
    big_m_load: float = 100_000.0
    big_m_battery: float = 1_000_000.0

def _slug(value: str) -> str:
    raw = str(value)
    compact = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_") or "id"
    digest = blake2b(raw.encode("utf-8"), digest_size=4).hexdigest()
    return f"{compact}_{digest}"

def _make_depot_copy_id(vehicle_id: str, depot_id: str, role: DepotRole, index: int | None) -> str:
    index_part = "none" if index is None else str(index)
    return f"__depotcopy__{_slug(vehicle_id)}__{_slug(depot_id)}__{role}__{index_part}"

def _vehicle_by_id(data: CECPDPTWData) -> dict[str, Vehicle]:
    return {v.id: v for v in data.vehicles}

def _customer_nodes(data: CECPDPTWData) -> list[str]:
    return [n for r in data.requests for n in (r.pickup, r.delivery)]

def _pickup_nodes(data: CECPDPTWData) -> set[str]:
    return {r.pickup for r in data.requests}

def _delivery_nodes(data: CECPDPTWData) -> set[str]:
    return {r.delivery for r in data.requests}

def _is_depot_copy(node: str, depot_meta: Mapping[str, DepotCopyMeta]) -> bool:
    return node in depot_meta

def _is_start_copy(node: str, depot_meta: Mapping[str, DepotCopyMeta]) -> bool:
    return node in depot_meta and depot_meta[node].role == "start"

def _is_mid_copy(node: str, depot_meta: Mapping[str, DepotCopyMeta]) -> bool:
    return node in depot_meta and depot_meta[node].role == "mid"

def _is_end_copy(node: str, depot_meta: Mapping[str, DepotCopyMeta]) -> bool:
    return node in depot_meta and depot_meta[node].role == "end"

def _physical_node(node: str, depot_meta: Mapping[str, DepotCopyMeta]) -> str:
    return depot_meta[node].physical_depot_id if node in depot_meta else node

def _fixed_key(vehicle_id: str, i: str, j: str, depot_meta: Mapping[str, DepotCopyMeta]) -> tuple[str, str, str]:
    return vehicle_id, _physical_node(i, depot_meta), _physical_node(j, depot_meta)

def _speed_key(
    vehicle_id: str,
    i: str,
    j: str,
    speed: str,
    depot_meta: Mapping[str, DepotCopyMeta],
) -> tuple[str, str, str, str]:
    return vehicle_id, _physical_node(i, depot_meta), _physical_node(j, depot_meta), speed

def _admissible_depots(data: CECPDPTWData, vehicle: Vehicle) -> list[str]:
    if data.depot_sharing:
        return list(data.physical_depots)
    return list(data.uav_depots if vehicle.mode == "UAV" else data.adr_depots)

def _validate_data(data: CECPDPTWData) -> None:
    vehicle_ids = [v.id for v in data.vehicles]
    request_ids = [r.id for r in data.requests]
    customer_ids = [node for r in data.requests for node in (r.pickup, r.delivery)]

    if len(vehicle_ids) != len(set(vehicle_ids)):
        raise ValueError("Vehicle IDs must be unique.")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Request IDs must be unique.")
    if len(customer_ids) != len(set(customer_ids)):
        raise ValueError("Pickup and delivery node IDs must be globally unique.")
    if not data.physical_depots:
        raise ValueError("At least one physical depot is required.")

    for vehicle in data.vehicles:
        if vehicle.capacity <= 0:
            raise ValueError(f"Vehicle {vehicle.id} must have positive capacity.")
        if vehicle.battery_max <= 0:
            raise ValueError(f"Vehicle {vehicle.id} must have positive battery_max.")
        if not 0 <= vehicle.battery_min <= vehicle.battery_max:
            raise ValueError(f"Vehicle {vehicle.id} has invalid battery_min.")
        if vehicle.depot_visit_limit < 0:
            raise ValueError(f"Vehicle {vehicle.id} has negative depot_visit_limit.")

    for request in data.requests:
        if request.demand <= 0:
            raise ValueError(f"Request {request.id} must have positive demand.")

def _build_depot_copies(
    data: CECPDPTWData,
) -> tuple[dict[str, list[str]], dict[str, DepotCopyMeta]]:
    copies_by_vehicle: dict[str, list[str]] = defaultdict(list)
    meta: dict[str, DepotCopyMeta] = {}

    for vehicle in data.vehicles:
        for depot_id in _admissible_depots(data, vehicle):
            start_id = _make_depot_copy_id(vehicle.id, depot_id, "start", 0)
            end_id = _make_depot_copy_id(vehicle.id, depot_id, "end", None)

            for copy_id, role, index in ((start_id, "start", 0), (end_id, "end", None)):
                copies_by_vehicle[vehicle.id].append(copy_id)
                meta[copy_id] = DepotCopyMeta(
                    copy_id=copy_id,
                    vehicle_id=vehicle.id,
                    physical_depot_id=depot_id,
                    role=role,
                    visit_index=index,
                )

            for h in range(1, vehicle.depot_visit_limit + 1):
                mid_id = _make_depot_copy_id(vehicle.id, depot_id, "mid", h)
                copies_by_vehicle[vehicle.id].append(mid_id)
                meta[mid_id] = DepotCopyMeta(
                    copy_id=mid_id,
                    vehicle_id=vehicle.id,
                    physical_depot_id=depot_id,
                    role="mid",
                    visit_index=h,
                )

    return dict(copies_by_vehicle), meta

def _adaptive_speed_levels_for_arc(
    data: CECPDPTWData,
    vehicle: Vehicle,
    i: str,
    j: str,
    depot_meta: Mapping[str, DepotCopyMeta],
    pickup_nodes: set[str],
    use_adaptive_speed: bool,
) -> tuple[str, ...]:
    if not use_adaptive_speed or j not in pickup_nodes or not vehicle.speed_levels:
        return ()

    valid_speeds = []
    for speed in vehicle.speed_levels:
        key = _speed_key(vehicle.id, i, j, speed, depot_meta)
        if key in data.pickup_time_by_speed and key in data.pickup_energy_by_speed:
            valid_speeds.append(speed)

    return tuple(valid_speeds) if len(valid_speeds) == len(vehicle.speed_levels) else ()

def _has_fixed_arc_cost(
    data: CECPDPTWData,
    vehicle_id: str,
    i: str,
    j: str,
    depot_meta: Mapping[str, DepotCopyMeta],
) -> bool:
    key = _fixed_key(vehicle_id, i, j, depot_meta)
    return key in data.travel_time and key in data.energy

def _build_nodes_and_arcs(
    data: CECPDPTWData,
    depot_copies_by_vehicle: dict[str, list[str]],
    depot_meta: dict[str, DepotCopyMeta],
    use_adaptive_speed: bool,
) -> tuple[dict[str, list[str]], list[tuple[str, str, str]], dict[tuple[str, str, str], tuple[str, ...]]]:
    customers = _customer_nodes(data)
    pickups = _pickup_nodes(data)

    nodes_by_vehicle: dict[str, list[str]] = {}
    arcs: list[tuple[str, str, str]] = []
    adaptive_arcs: dict[tuple[str, str, str], tuple[str, ...]] = {}

    for vehicle in data.vehicles:
        nodes = customers + depot_copies_by_vehicle[vehicle.id]
        nodes_by_vehicle[vehicle.id] = nodes

        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                if _is_end_copy(i, depot_meta) or _is_start_copy(j, depot_meta):
                    continue
                if _is_depot_copy(i, depot_meta) and _is_depot_copy(j, depot_meta):
                    continue

                speeds = _adaptive_speed_levels_for_arc(
                    data=data,
                    vehicle=vehicle,
                    i=i,
                    j=j,
                    depot_meta=depot_meta,
                    pickup_nodes=pickups,
                    use_adaptive_speed=use_adaptive_speed,
                )

                if speeds:
                    arc = (vehicle.id, i, j)
                    arcs.append(arc)
                    adaptive_arcs[arc] = speeds
                    continue

                if _has_fixed_arc_cost(data, vehicle.id, i, j, depot_meta):
                    arcs.append((vehicle.id, i, j))

    return nodes_by_vehicle, arcs, adaptive_arcs

def build_ce_cpdptw_model(
    data: CECPDPTWData,
    *,
    use_adaptive_speed: bool = True,
    tighten_battery_equalities: bool = True,
    model_name: str = "CE_CPDPTW",
) -> BuiltCECPDPTWModel:
    _validate_data(data)

    vehicles = _vehicle_by_id(data)
    vehicle_ids = [v.id for v in data.vehicles]
    request_ids = [r.id for r in data.requests]

    customer_nodes = set(_customer_nodes(data))
    depot_copies_by_vehicle, depot_meta = _build_depot_copies(data)

    nodes_by_vehicle, arcs, adaptive_arcs = _build_nodes_and_arcs(
        data=data,
        depot_copies_by_vehicle=depot_copies_by_vehicle,
        depot_meta=depot_meta,
        use_adaptive_speed=use_adaptive_speed,
    )

    if data.requests and not arcs:
        raise ValueError("No feasible arcs were generated. Check distances, masks, and depot data.")

    out_nodes: dict[tuple[str, str], list[str]] = defaultdict(list)
    in_nodes: dict[tuple[str, str], list[str]] = defaultdict(list)

    for k, i, j in arcs:
        out_nodes[k, i].append(j)
        in_nodes[k, j].append(i)

    node_keys = [(k, i) for k in vehicle_ids for i in nodes_by_vehicle[k]]
    depot_keys = [(k, i) for k in vehicle_ids for i in depot_copies_by_vehicle[k]]
    request_vehicle_keys = [(r.id, k) for r in data.requests for k in vehicle_ids]
    z_keys = [(k, i, j, speed) for (k, i, j), speeds in adaptive_arcs.items() for speed in speeds]

    model = gp.Model(model_name)

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    z = model.addVars(z_keys, vtype=GRB.BINARY, name="z") if z_keys else None

    use_vehicle = model.addVars(vehicle_ids, vtype=GRB.BINARY, name="use_vehicle")
    depot_visit = model.addVars(depot_keys, vtype=GRB.BINARY, name="depot_visit")

    arrival = model.addVars(node_keys, lb=0.0, name="arrival")
    completion = model.addVars(node_keys, lb=0.0, name="completion")
    waiting = model.addVars(node_keys, lb=0.0, name="waiting")
    load = model.addVars(node_keys, lb=0.0, name="load")
    battery = model.addVars(node_keys, lb=0.0, name="battery")
    battery_slack = model.addVars(node_keys, lb=0.0, name="battery_slack")

    early_pickup = model.addVars(request_vehicle_keys, lb=0.0, name="early_pickup")
    late_pickup = model.addVars(request_vehicle_keys, lb=0.0, name="late_pickup")
    late_delivery = model.addVars(request_vehicle_keys, lb=0.0, name="late_delivery")

    y_uav = model.addVars(request_ids, vtype=GRB.BINARY, name="y_uav")
    y_adr = model.addVars(request_ids, vtype=GRB.BINARY, name="y_adr")

    def out_sum(k: str, i: str) -> gp.LinExpr:
        return gp.quicksum(x[k, i, j] for j in out_nodes.get((k, i), []))

    def in_sum(k: str, i: str) -> gp.LinExpr:
        return gp.quicksum(x[k, j, i] for j in in_nodes.get((k, i), []))

    def served_expr(request: Request, k: str) -> gp.LinExpr:
        return out_sum(k, request.pickup)

    def node_demand(node: str) -> float:
        for request in data.requests:
            if node == request.pickup:
                return request.demand
            if node == request.delivery:
                return -request.demand
        return 0.0

    def service_time(k: str, node: str) -> float:
        if node not in customer_nodes:
            return 0.0
        return data.uav_customer_service_time if vehicles[k].mode == "UAV" else data.adr_customer_service_time

    def recharge_duration(k: str) -> float:
        return data.uav_recharge_duration if vehicles[k].mode == "UAV" else data.adr_recharge_duration

    def fixed_time(k: str, i: str, j: str) -> float:
        return data.travel_time[_fixed_key(k, i, j, depot_meta)]

    def fixed_energy(k: str, i: str, j: str) -> float:
        return data.energy[_fixed_key(k, i, j, depot_meta)]

    def travel_expr(k: str, i: str, j: str, *, multiply_by_x: bool) -> Any:
        arc = (k, i, j)
        if z is not None and arc in adaptive_arcs:
            return gp.quicksum(
                data.pickup_time_by_speed[_speed_key(k, i, j, speed, depot_meta)] * z[k, i, j, speed]
                for speed in adaptive_arcs[arc]
            )
        value = fixed_time(k, i, j)
        return value * x[k, i, j] if multiply_by_x else value

    def energy_expr(k: str, i: str, j: str, *, multiply_by_x: bool) -> Any:
        arc = (k, i, j)
        if z is not None and arc in adaptive_arcs:
            return gp.quicksum(
                data.pickup_energy_by_speed[_speed_key(k, i, j, speed, depot_meta)] * z[k, i, j, speed]
                for speed in adaptive_arcs[arc]
            )
        value = fixed_energy(k, i, j)
        return value * x[k, i, j] if multiply_by_x else value

    travel_cost = gp.quicksum(
        (data.alpha_uav if vehicles[k].mode == "UAV" else data.alpha_adr)
        * travel_expr(k, i, j, multiply_by_x=True)
        for k, i, j in arcs
    )

    penalty_cost = (
        data.alpha_early_pickup * gp.quicksum(early_pickup[r, k] for r, k in request_vehicle_keys)
        + data.alpha_late_pickup * gp.quicksum(late_pickup[r, k] for r, k in request_vehicle_keys)
        + data.alpha_late_delivery * gp.quicksum(late_delivery[r, k] for r, k in request_vehicle_keys)
        + data.lambda_battery * gp.quicksum(battery_slack[k, i] for k, i in node_keys)
    )

    model.setObjective(travel_cost + penalty_cost, GRB.MINIMIZE)

    if z is not None:
        for k, i, j in adaptive_arcs:
            model.addConstr(
                gp.quicksum(z[k, i, j, speed] for speed in adaptive_arcs[k, i, j]) == x[k, i, j],
                name=f"choose_speed[{k},{i},{j}]",
            )

    for k in vehicle_ids:
        start_copies = [i for i in depot_copies_by_vehicle[k] if _is_start_copy(i, depot_meta)]
        end_copies = [i for i in depot_copies_by_vehicle[k] if _is_end_copy(i, depot_meta)]
        mid_copies = [i for i in depot_copies_by_vehicle[k] if _is_mid_copy(i, depot_meta)]

        model.addConstr(
            gp.quicksum(out_sum(k, i) for i in start_copies) == use_vehicle[k],
            name=f"start_once_if_used[{k}]",
        )
        model.addConstr(
            gp.quicksum(in_sum(k, i) for i in end_copies) == use_vehicle[k],
            name=f"end_once_if_used[{k}]",
        )

        served_count = gp.quicksum(served_expr(r, k) for r in data.requests)
        model.addConstr(served_count <= len(data.requests) * use_vehicle[k], name=f"use_upper[{k}]")
        model.addConstr(served_count >= use_vehicle[k], name=f"use_lower[{k}]")

        for node in customer_nodes:
            model.addConstr(out_sum(k, node) == in_sum(k, node), name=f"flow_customer[{k},{node}]")
            model.addConstr(out_sum(k, node) <= 1, name=f"customer_out_once[{k},{node}]")
            model.addConstr(in_sum(k, node) <= 1, name=f"customer_in_once[{k},{node}]")

        for node in mid_copies:
            model.addConstr(out_sum(k, node) == in_sum(k, node), name=f"flow_mid_depot[{k},{node}]")
            model.addConstr(out_sum(k, node) <= 1, name=f"mid_depot_out_once[{k},{node}]")
            model.addConstr(in_sum(k, node) <= 1, name=f"mid_depot_in_once[{k},{node}]")

        for node in start_copies:
            model.addConstr(in_sum(k, node) == 0, name=f"no_in_start[{k},{node}]")
            model.addConstr(depot_visit[k, node] == 0, name=f"start_no_recharge[{k},{node}]")

        for node in end_copies:
            model.addConstr(out_sum(k, node) == 0, name=f"no_out_end[{k},{node}]")

    for request in data.requests:
        model.addConstr(
            gp.quicksum(served_expr(request, k) for k in vehicle_ids) == 1,
            name=f"pickup_served_once[{request.id}]",
        )
        model.addConstr(
            gp.quicksum(out_sum(k, request.delivery) for k in vehicle_ids) == 1,
            name=f"delivery_served_once[{request.id}]",
        )

        for k in vehicle_ids:
            model.addConstr(
                served_expr(request, k) == out_sum(k, request.delivery),
                name=f"same_vehicle[{request.id},{k}]",
            )

    for k, depot_copy in depot_keys:
        if _is_start_copy(depot_copy, depot_meta):
            continue
        model.addConstr(
            depot_visit[k, depot_copy] == in_sum(k, depot_copy),
            name=f"depot_activation[{k},{depot_copy}]",
        )

    for k in vehicle_ids:
        vehicle = vehicles[k]

        for node in nodes_by_vehicle[k]:
            model.addConstr(load[k, node] <= vehicle.capacity, name=f"load_bound[{k},{node}]")
            model.addConstr(battery[k, node] <= vehicle.battery_max, name=f"battery_bound[{k},{node}]")
            model.addConstr(
                battery_slack[k, node] >= vehicle.battery_min - battery[k, node],
                name=f"battery_threshold_slack[{k},{node}]",
            )

            if node in customer_nodes:
                visit = out_sum(k, node)
                model.addConstr(arrival[k, node] <= data.big_m_time * visit, name=f"A_zero_unused[{k},{node}]")
                model.addConstr(completion[k, node] <= data.big_m_time * visit, name=f"T_zero_unused[{k},{node}]")
                model.addConstr(waiting[k, node] <= data.big_m_time * visit, name=f"W_zero_unused[{k},{node}]")
                model.addConstr(
                    completion[k, node] == arrival[k, node] + waiting[k, node] + service_time(k, node) * visit,
                    name=f"completion_customer[{k},{node}]",
                )
            else:
                model.addConstr(waiting[k, node] == 0, name=f"no_wait_depot[{k},{node}]")
                model.addConstr(completion[k, node] == arrival[k, node], name=f"completion_depot[{k},{node}]")

                if _is_start_copy(node, depot_meta):
                    model.addConstr(arrival[k, node] == 0, name=f"start_arrival_zero[{k},{node}]")
                    model.addConstr(completion[k, node] == 0, name=f"start_completion_zero[{k},{node}]")
                    model.addConstr(load[k, node] == 0, name=f"start_load_zero[{k},{node}]")
                    model.addConstr(battery[k, node] == vehicle.battery_max, name=f"start_battery_full[{k},{node}]")
                else:
                    model.addConstr(
                        arrival[k, node] <= data.big_m_time * depot_visit[k, node],
                        name=f"depot_A_zero_unused[{k},{node}]",
                    )
                    model.addConstr(
                        completion[k, node] <= data.big_m_time * depot_visit[k, node],
                        name=f"depot_T_zero_unused[{k},{node}]",
                    )
                    model.addConstr(
                        battery[k, node] >= vehicle.battery_max - data.big_m_battery * (1 - depot_visit[k, node]),
                        name=f"depot_recharge[{k},{node}]",
                    )

    for k, i, j in arcs:
        recharge_wait = (
            recharge_duration(k) * depot_visit[k, j]
            if _is_depot_copy(j, depot_meta) and not _is_start_copy(j, depot_meta)
            else 0.0
        )

        model.addConstr(
            arrival[k, j]
            >= completion[k, i]
            + travel_expr(k, i, j, multiply_by_x=False)
            + recharge_wait
            - data.big_m_time * (1 - x[k, i, j]),
            name=f"time_prop[{k},{i},{j}]",
        )

        model.addConstr(
            load[k, j] >= load[k, i] + node_demand(j) - data.big_m_load * (1 - x[k, i, j]),
            name=f"load_lb[{k},{i},{j}]",
        )
        model.addConstr(
            load[k, j] <= load[k, i] + node_demand(j) + data.big_m_load * (1 - x[k, i, j]),
            name=f"load_ub[{k},{i},{j}]",
        )

        if not _is_depot_copy(j, depot_meta):
            model.addConstr(
                battery[k, j]
                <= battery[k, i]
                - energy_expr(k, i, j, multiply_by_x=False)
                + vehicles[k].battery_max * (1 - x[k, i, j]),
                name=f"battery_non_depot_ub[{k},{i},{j}]",
            )

            if tighten_battery_equalities:
                model.addConstr(
                    battery[k, j]
                    >= battery[k, i]
                    - energy_expr(k, i, j, multiply_by_x=False)
                    - vehicles[k].battery_max * (1 - x[k, i, j]),
                    name=f"battery_non_depot_lb[{k},{i},{j}]",
                )

        if i in customer_nodes and _is_depot_copy(j, depot_meta):
            model.addConstr(
                battery[k, i]
                - energy_expr(k, i, j, multiply_by_x=False)
                >= -data.big_m_battery * (1 - x[k, i, j]),
                name=f"return_to_depot_battery[{k},{i},{j}]",
            )

    for request in data.requests:
        for k in vehicle_ids:
            served = served_expr(request, k)

            model.addConstr(
                waiting[k, request.pickup]
                >= request.pickup_target - arrival[k, request.pickup] - data.big_m_time * (1 - served),
                name=f"pickup_wait[{request.id},{k}]",
            )

            model.addConstr(
                arrival[k, request.delivery]
                >= completion[k, request.pickup] - data.big_m_time * (1 - served),
                name=f"precedence[{request.id},{k}]",
            )

            model.addConstr(
                early_pickup[request.id, k]
                >= request.pickup_target - arrival[k, request.pickup] - data.big_m_time * (1 - served),
                name=f"early_pickup_lb[{request.id},{k}]",
            )
            model.addConstr(
                late_pickup[request.id, k]
                >= arrival[k, request.pickup] - request.pickup_latest - data.big_m_time * (1 - served),
                name=f"late_pickup_lb[{request.id},{k}]",
            )
            model.addConstr(
                late_delivery[request.id, k]
                >= completion[k, request.delivery] - request.delivery_latest - data.big_m_time * (1 - served),
                name=f"late_delivery_lb[{request.id},{k}]",
            )

            model.addConstr(early_pickup[request.id, k] <= data.big_m_time * served)
            model.addConstr(late_pickup[request.id, k] <= data.big_m_time * served)
            model.addConstr(late_delivery[request.id, k] <= data.big_m_time * served)

    uav_vehicle_ids = [v.id for v in data.vehicles if v.mode == "UAV"]
    adr_vehicle_ids = [v.id for v in data.vehicles if v.mode == "ADR"]

    for request in data.requests:
        model.addConstr(
            y_uav[request.id] == gp.quicksum(served_expr(request, k) for k in uav_vehicle_ids),
            name=f"mode_uav[{request.id}]",
        )
        model.addConstr(
            y_adr[request.id] == gp.quicksum(served_expr(request, k) for k in adr_vehicle_ids),
            name=f"mode_adr[{request.id}]",
        )
        model.addConstr(y_uav[request.id] + y_adr[request.id] == 1, name=f"mode_partition[{request.id}]")

    shared_requests = [r for r in data.requests if r.market == "shared"]
    uav_only_requests = [r for r in data.requests if r.market == "UAV"]
    adr_only_requests = [r for r in data.requests if r.market == "ADR"]

    model.addConstr(
        gp.quicksum(y_adr[r.id] for r in shared_requests if r.owner_mode == "UAV")
        + gp.quicksum(y_uav[r.id] for r in shared_requests if r.owner_mode == "ADR")
        <= data.epsilon_shared * len(shared_requests),
        name="shared_market_cap",
    )
    model.addConstr(
        gp.quicksum(y_adr[r.id] for r in uav_only_requests)
        <= data.epsilon_to_adr * len(uav_only_requests),
        name="uav_to_adr_cap",
    )
    model.addConstr(
        gp.quicksum(y_uav[r.id] for r in adr_only_requests)
        <= data.epsilon_to_uav * len(adr_only_requests),
        name="adr_to_uav_cap",
    )

    model.update()

    return BuiltCECPDPTWModel(
        model=model,
        data=data,
        nodes_by_vehicle=nodes_by_vehicle,
        depot_meta=depot_meta,
        depot_copies_by_vehicle=depot_copies_by_vehicle,
        arcs=arcs,
        adaptive_arcs=adaptive_arcs,
        x=x,
        z=z,
        use_vehicle=use_vehicle,
        depot_visit=depot_visit,
        arrival=arrival,
        completion=completion,
        waiting=waiting,
        load=load,
        battery=battery,
        early_pickup=early_pickup,
        late_pickup=late_pickup,
        late_delivery=late_delivery,
        battery_slack=battery_slack,
        y_uav=y_uav,
        y_adr=y_adr,
    )

def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def select_instance(instance_or_batch: Mapping[str, Any], batch_index: int = 0) -> dict[str, Any]:
    x = to_numpy(instance_or_batch["x"])

    if x.ndim != 3:
        return dict(instance_or_batch)

    selected: dict[str, Any] = {}

    for key, value in instance_or_batch.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                selected[key] = value
            elif key in {"n_depots", "n_req"}:
                selected[key] = int(value[batch_index].item())
            else:
                selected[key] = value[batch_index]
            continue

        arr = np.asarray(value)
        if arr.ndim == 0:
            selected[key] = value
        elif key in {"n_depots", "n_req"}:
            selected[key] = int(arr[batch_index].item())
        else:
            selected[key] = arr[batch_index]

    return selected

def build_ce_cpdptw_data_from_synthetic(
    instance: Mapping[str, Any],
    config: SyntheticAdapterConfig,
) -> CECPDPTWData:
    n_depots = int(to_numpy(instance["n_depots"]).item())
    n_req = int(to_numpy(instance["n_req"]).item())
    n_total = n_depots + 2 * n_req

    expected_depots = config.n_depots_uav + config.n_depots_adr
    if expected_depots != n_depots:
        raise ValueError(f"Depot split mismatch: config={expected_depots}, instance={n_depots}")

    expected_agents = config.n_uav + config.n_adr
    capacity = to_numpy(instance["capacity"]).reshape(-1).astype(float)
    battery = to_numpy(instance["battery"]).reshape(-1).astype(float)

    if len(capacity) != expected_agents:
        raise ValueError(f"Capacity length mismatch: expected {expected_agents}, got {len(capacity)}")
    if len(battery) != expected_agents:
        raise ValueError(f"Battery length mismatch: expected {expected_agents}, got {len(battery)}")

    node_features = to_numpy(instance["x"]).reshape(n_total, -1).astype(float)
    demand = to_numpy(instance["demand"]).reshape(n_total).astype(float)
    time_window = to_numpy(instance["time_window"]).reshape(n_total).astype(float)

    dist_uav = to_numpy(instance["edge_attr_d"]).reshape(n_total, n_total).astype(float)
    dist_adr = to_numpy(instance["edge_attr_r"]).reshape(n_total, n_total).astype(float)

    mask_uav = to_numpy(instance["mask_adjacency_uav"]).reshape(n_total, n_total) > 0.5
    mask_adr = to_numpy(instance["mask_adjacency_adr"]).reshape(n_total, n_total) > 0.5

    acc_uav = node_features[:, 8] > 0.5
    acc_adr = node_features[:, 9] > 0.5

    node_ids = [f"n{i}" for i in range(n_total)]
    physical_depots = node_ids[:n_depots]
    uav_depots = node_ids[:config.n_depots_uav]
    adr_depots = node_ids[config.n_depots_uav:n_depots]

    requests = _build_requests_from_synthetic(
        n_depots=n_depots,
        n_req=n_req,
        node_ids=node_ids,
        demand=demand,
        time_window=time_window,
        acc_uav=acc_uav,
        acc_adr=acc_adr,
        config=config,
    )

    vehicles = _build_vehicles_from_synthetic(capacity=capacity, battery=battery, config=config)

    travel_time: dict[tuple[str, str, str], float] = {}
    energy: dict[tuple[str, str, str], float] = {}
    pickup_time_by_speed: dict[tuple[str, str, str, str], float] = {}
    pickup_energy_by_speed: dict[tuple[str, str, str, str], float] = {}

    for vehicle in vehicles:
        if vehicle.mode == "UAV":
            dist = dist_uav
            mask = mask_uav
            access = acc_uav
            cruise_speed = V_UAV_MAX
            depot_speed = V_UAV_DEPOT
            min_pickup_speed = V_UAV_MIN_PICKUP
            power_fn = pc_uav_numpy
        else:
            dist = dist_adr
            mask = mask_adr
            access = acc_adr
            cruise_speed = V_ADR_MAX
            depot_speed = V_ADR_DEPOT
            min_pickup_speed = V_ADR_MIN_PICKUP
            power_fn = pc_adr_numpy

        if config.pickup_speed_grid_size < 2:
            raise ValueError("pickup_speed_grid_size must be at least 2.")

        pickup_speed_grid = np.linspace(min_pickup_speed, cruise_speed, config.pickup_speed_grid_size)

        for i in range(n_total):
            for j in range(n_total):
                if not _valid_physical_arc(
                    i=i,
                    j=j,
                    n_depots=n_depots,
                    distance_matrix=dist,
                    mask=mask,
                    access=access,
                    require_destination_access=config.require_destination_access,
                ):
                    continue

                source = node_ids[i]
                target = node_ids[j]
                distance = float(dist[i, j])

                speed = depot_speed if _is_depot_return(i, j, n_depots) else cruise_speed
                payload = _target_payload_for_energy(
                    j=j,
                    n_depots=n_depots,
                    n_req=n_req,
                    demand=demand,
                    capacity=vehicle.capacity,
                    is_depot_return=_is_depot_return(i, j, n_depots),
                    depot_policy=config.depot_return_energy_payload_policy,
                )

                travel_time[vehicle.id, source, target] = distance / speed
                energy[vehicle.id, source, target] = energy_joules_from_power(
                    distance=distance,
                    speed_coord_per_min=speed,
                    payload_kg=payload,
                    power_fn=power_fn,
                )

                if config.use_discrete_pickup_speed and _is_pickup_index(j, n_depots, n_req):
                    pickup_payload = max(0.0, float(demand[j]))
                    for speed_value in pickup_speed_grid:
                        speed_key = f"{speed_value:.6g}"
                        pickup_time_by_speed[vehicle.id, source, target, speed_key] = distance / speed_value
                        pickup_energy_by_speed[vehicle.id, source, target, speed_key] = energy_joules_from_power(
                            distance=distance,
                            speed_coord_per_min=float(speed_value),
                            payload_kg=pickup_payload,
                            power_fn=power_fn,
                        )

    return CECPDPTWData(
        requests=requests,
        vehicles=vehicles,
        physical_depots=physical_depots,
        uav_depots=uav_depots,
        adr_depots=adr_depots,
        travel_time=travel_time,
        energy=energy,
        pickup_time_by_speed=pickup_time_by_speed,
        pickup_energy_by_speed=pickup_energy_by_speed,
        depot_sharing=config.depot_sharing,
        alpha_uav=config.alpha_uav,
        alpha_adr=config.alpha_adr,
        alpha_early_pickup=config.alpha_early_pickup,
        alpha_late_pickup=config.alpha_late_pickup,
        alpha_late_delivery=config.alpha_late_delivery,
        lambda_battery=config.lambda_battery,
        epsilon_shared=config.epsilon_shared,
        epsilon_to_adr=config.epsilon_to_adr,
        epsilon_to_uav=config.epsilon_to_uav,
        uav_customer_service_time=UAV_LAND_TAKEOFF_MIN,
        adr_customer_service_time=ADR_CUSTOMER_SERVICE_MIN,
        uav_recharge_duration=UAV_RECHARGE_MIN,
        adr_recharge_duration=ADR_RECHARGE_MIN,
        big_m_time=config.big_m_time,
        big_m_load=config.big_m_load,
        big_m_battery=config.big_m_battery,
    )

def _build_requests_from_synthetic(
    *,
    n_depots: int,
    n_req: int,
    node_ids: list[str],
    demand: np.ndarray,
    time_window: np.ndarray,
    acc_uav: np.ndarray,
    acc_adr: np.ndarray,
    config: SyntheticAdapterConfig,
) -> list[Request]:
    requests: list[Request] = []

    for r in range(n_req):
        pickup_idx = n_depots + r
        delivery_idx = n_depots + n_req + r

        can_uav = bool(acc_uav[pickup_idx] and acc_uav[delivery_idx])
        can_adr = bool(acc_adr[pickup_idx] and acc_adr[delivery_idx])

        if not can_uav and not can_adr and config.strict_pair_access:
            raise ValueError(f"Request r{r} has no single feasible mode for both pickup and delivery.")

        if can_uav and can_adr:
            market: Market = "shared"
            owner_mode: Mode | None = None
        elif can_uav:
            market = "UAV"
            owner_mode = "UAV"
        elif can_adr:
            market = "ADR"
            owner_mode = "ADR"
        else:
            market = "shared"
            owner_mode = None

        pickup_target = float(time_window[pickup_idx])
        delivery_target = float(time_window[delivery_idx])

        requests.append(
            Request(
                id=f"r{r}",
                pickup=node_ids[pickup_idx],
                delivery=node_ids[delivery_idx],
                demand=abs(float(demand[pickup_idx])),
                pickup_target=pickup_target,
                pickup_latest=pickup_target + config.latest_pickup_slack,
                delivery_target=delivery_target,
                delivery_latest=delivery_target + config.latest_delivery_slack,
                market=market,
                owner_mode=owner_mode,
            )
        )

    return requests

def _build_vehicles_from_synthetic(
    *,
    capacity: np.ndarray,
    battery: np.ndarray,
    config: SyntheticAdapterConfig,
) -> list[Vehicle]:
    vehicles: list[Vehicle] = []
    depot_visit_limit = config.depot_visit_limit if config.depot_visit_limit is not None else max(1, config.n_uav + config.n_adr)

    uav_speed_keys = tuple(f"{v:.6g}" for v in np.linspace(V_UAV_MIN_PICKUP, V_UAV_MAX, config.pickup_speed_grid_size))
    adr_speed_keys = tuple(f"{v:.6g}" for v in np.linspace(V_ADR_MIN_PICKUP, V_ADR_MAX, config.pickup_speed_grid_size))

    for i in range(config.n_uav):
        vehicles.append(
            Vehicle(
                id=f"uav_{i}",
                mode="UAV",
                capacity=float(capacity[i]),
                battery_max=float(battery[i]),
                battery_min=float(UAV_LOW_BATTERY_FRAC * battery[i]),
                depot_visit_limit=depot_visit_limit,
                speed_levels=uav_speed_keys if config.use_discrete_pickup_speed else (),
            )
        )

    offset = config.n_uav
    for i in range(config.n_adr):
        idx = offset + i
        vehicles.append(
            Vehicle(
                id=f"adr_{i}",
                mode="ADR",
                capacity=float(capacity[idx]),
                battery_max=float(battery[idx]),
                battery_min=float(ADR_LOW_BATTERY_FRAC * battery[idx]),
                depot_visit_limit=depot_visit_limit,
                speed_levels=adr_speed_keys if config.use_discrete_pickup_speed else (),
            )
        )

    return vehicles

def pc_uav_numpy(payload_kg: float, v_ground_mps: float) -> float:
    nu = 0.9
    rho = 1.225
    mass = 12.0
    c_d1 = 1.49
    c_d2 = 2.2
    a1 = 0.224
    a2 = 0.1
    v_w = 0.0

    v_a = math.sqrt(max(v_ground_mps**2 + v_w**2, 1e-8))
    drag = rho * (c_d1 * a1 + c_d2 * a2) * v_a**2 / 2.0
    weight = 9.8 * (mass + payload_kg)
    thrust = drag + weight
    alpha = math.atan2(drag, weight)
    vi = 1.0

    power_kw = thrust * (v_a * math.sin(alpha) + vi) / nu / 1000.0
    return max(0.0, power_kw)

def pc_adr_numpy(payload_kg: float, v_ground_mps: float) -> float:
    c_r = 0.25
    nu = 0.8
    mass = 30.0

    power_kw = c_r * (mass + payload_kg) * 9.8 * v_ground_mps / nu / 1000.0
    return max(0.0, power_kw)

def energy_joules_from_power(
    *,
    distance: float,
    speed_coord_per_min: float,
    payload_kg: float,
    power_fn,
) -> float:
    if distance <= 1e-9:
        return 0.0

    speed_mps = speed_coord_per_min * SCALE_M_PER_COORD / 60.0
    travel_minutes = distance / speed_coord_per_min
    power_kw = power_fn(payload_kg, speed_mps)

    return travel_minutes * power_kw * 60.0

def _target_payload_for_energy(
    *,
    j: int,
    n_depots: int,
    n_req: int,
    demand: np.ndarray,
    capacity: float,
    is_depot_return: bool,
    depot_policy: Literal["zero", "max_capacity"],
) -> float:
    if is_depot_return:
        return 0.0 if depot_policy == "zero" else capacity

    if _is_pickup_index(j, n_depots, n_req):
        return max(0.0, float(demand[j]))

    if _is_delivery_index(j, n_depots, n_req):
        return max(0.0, -float(demand[j]))

    return 0.0

def _valid_physical_arc(
    *,
    i: int,
    j: int,
    n_depots: int,
    distance_matrix: np.ndarray,
    mask: np.ndarray,
    access: np.ndarray,
    require_destination_access: bool,
) -> bool:
    if i == j:
        return False
    if i < n_depots and j < n_depots:
        return False
    if not np.isfinite(distance_matrix[i, j]) or distance_matrix[i, j] >= 1e8:
        return False
    if not bool(mask[i, j]):
        return False
    if require_destination_access and not bool(access[j]):
        return False
    return True

def _is_pickup_index(index: int, n_depots: int, n_req: int) -> bool:
    return n_depots <= index < n_depots + n_req

def _is_delivery_index(index: int, n_depots: int, n_req: int) -> bool:
    return n_depots + n_req <= index < n_depots + 2 * n_req

def _is_depot_return(i: int, j: int, n_depots: int) -> bool:
    return i >= n_depots and j < n_depots

def solve_ce_cpdptw(
    data: CECPDPTWData,
    *,
    use_adaptive_speed: bool = True,
    tighten_battery_equalities: bool = True,
    time_limit_seconds: float | None = 3600.0,
    mip_gap: float | None = 0.01,
    log_to_console: bool = True,
    n_threads: int = 0,
) -> BuiltCECPDPTWModel:
    built = build_ce_cpdptw_model(
        data,
        use_adaptive_speed=use_adaptive_speed,
        tighten_battery_equalities=tighten_battery_equalities,
    )

    model = built.model
    model.Params.LogToConsole = 1 if log_to_console else 0
    if n_threads > 0:
        model.Params.Threads = n_threads

    if time_limit_seconds is not None:
        model.Params.TimeLimit = time_limit_seconds
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    model.optimize()
    return built

def solve_synthetic_instance(
    instance: Mapping[str, Any],
    config: SyntheticAdapterConfig,
    *,
    batch_index: int = 0,
    time_limit_seconds: float | None = 3600.0,
    mip_gap: float | None = 0.01,
    log_to_console: bool = True,
) -> tuple[BuiltCECPDPTWModel, dict[str, Any]]:
    single_instance = select_instance(instance, batch_index=batch_index)
    data = build_ce_cpdptw_data_from_synthetic(single_instance, config)

    built = solve_ce_cpdptw(
        data,
        use_adaptive_speed=config.use_discrete_pickup_speed,
        time_limit_seconds=time_limit_seconds,
        mip_gap=mip_gap,
        log_to_console=log_to_console,
    )

    if built.model.Status in {GRB.INFEASIBLE, GRB.INF_OR_UNBD}:
        write_debug_files(built)

    return built, extract_solution(built)

def _display_node(node: str, depot_meta: Mapping[str, DepotCopyMeta]) -> str:
    if node not in depot_meta:
        return node

    meta = depot_meta[node]
    if meta.role == "mid":
        return f"depot({meta.physical_depot_id}, visit={meta.visit_index})"
    return f"depot({meta.physical_depot_id}, {meta.role})"

def extract_solution(built: BuiltCECPDPTWModel, *, tolerance: float = 0.5) -> dict[str, Any]:
    model = built.model

    result: dict[str, Any] = {
        "status": int(model.Status),
        "status_name": _status_name(model.Status),
        "objective": None,
        "routes": {},
        "served_requests": {},
        "timing": {},
        "battery": {},
        "load": {},
    }

    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} or model.SolCount == 0:
        return result

    result["objective"] = float(model.ObjVal)

    successors: dict[tuple[str, str], str] = {}
    active_arcs: list[tuple[str, str, str]] = []

    for k, i, j in built.arcs:
        if built.x[k, i, j].X >= tolerance:
            successors[k, i] = j
            active_arcs.append((k, i, j))

    for vehicle in built.data.vehicles:
        k = vehicle.id
        route: list[str] = []

        start = None
        for depot_copy in built.depot_copies_by_vehicle[k]:
            if built.depot_meta[depot_copy].role == "start" and (k, depot_copy) in successors:
                start = depot_copy
                break

        if start is None:
            result["routes"][k] = []
            continue

        current = start
        seen = set()

        while current is not None and current not in seen:
            seen.add(current)
            route.append(_display_node(current, built.depot_meta))
            result["timing"][(k, _display_node(current, built.depot_meta))] = {
                "arrival": float(built.arrival[k, current].X),
                "completion": float(built.completion[k, current].X),
            }
            result["battery"][(k, _display_node(current, built.depot_meta))] = float(built.battery[k, current].X)
            result["load"][(k, _display_node(current, built.depot_meta))] = float(built.load[k, current].X)
            current = successors.get((k, current))

        if current is not None:
            route.append(f"cycle_detected_at({_display_node(current, built.depot_meta)})")

        result["routes"][k] = route

    for request in built.data.requests:
        if built.y_uav[request.id].X >= tolerance:
            result["served_requests"][request.id] = "UAV"
        elif built.y_adr[request.id].X >= tolerance:
            result["served_requests"][request.id] = "ADR"
        else:
            result["served_requests"][request.id] = "unassigned"

    return result

def write_debug_files(built: BuiltCECPDPTWModel, *, prefix: str = "ce_cpdptw_debug") -> None:
    built.model.write(f"{prefix}.lp")

    if built.model.Status == GRB.INFEASIBLE:
        built.model.computeIIS()
        built.model.write(f"{prefix}.ilp")

def _status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return names.get(status, f"STATUS_{status}")

def demo_from_dataset_module() -> None:
    from creat_vrp import create_instance

    rng = np.random.default_rng(42)

    instance = create_instance(
        n_req=8,
        n_uav=2,
        n_adr=2,
        n_depots_uav=1,
        n_depots_adr=1,
        rng=rng,
    )

    config = SyntheticAdapterConfig(
        n_uav=2,
        n_adr=2,
        n_depots_uav=1,
        n_depots_adr=1,
        depot_visit_limit=8,
        use_discrete_pickup_speed=True,
        pickup_speed_grid_size=5,
        latest_pickup_slack=0.0,
        latest_delivery_slack=0.0,
        lambda_battery=0.0,
    )

    _, solution = solve_synthetic_instance(
        instance,
        config,
        time_limit_seconds=600,
        mip_gap=0.01,
        log_to_console=True,
    )

    print(solution)

if __name__ == "__main__":
    demo_from_dataset_module()
