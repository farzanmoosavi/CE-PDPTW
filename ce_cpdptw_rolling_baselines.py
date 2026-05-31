from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ce_cpdptw_alns import (
    ALNSBaseline,
    Leg,
    Request,
    RollingHorizonDispatcher,
    SyntheticALNSConfig,
    Vehicle,
    _battery_threshold,
    _cruise_speed,
    _depot_speed,
    _edge_energy_kj,
    _get_distance_matrix,
    _is_arc_feasible,
    _nearest_feasible_depot,
    _pickup_speed,
    _service_time,
    _simulate_leg,
    _to_numpy,
    _wind_vec,
    build_initial_fleet_from_instance,
    build_dynamic_arrival_stream_from_instance,
    build_toy_full_instance,
    solve_static_instance_with_alns,
    summarize_episode_log,
)

@dataclass(frozen=True)
class ExactRollingConfig:
    n_uav: int
    n_adr: int
    n_depots_uav: int
    n_depots_adr: int
    depot_sharing: bool = True
    depot_visit_limit: int = 4
    delta_minutes: float = 5.0
    shift_minutes: Optional[float] = None
    latest_pickup_slack: float = 0.0
    latest_delivery_slack: float = 0.0
    use_discrete_pickup_speed: bool = False
    pickup_speed_grid_size: int = 5
    time_limit_seconds: float = 3600.0
    mip_gap: float = 0.01
    log_to_console: bool = False
    n_threads: int = 0
    # When True: zero all energy values and remove depot copies, turning the MILP
    # into a pure PDPTW. Routes are still executed against real battery physics.
    # Used for the ablation: (gurobi - gurobi_no_battery) = value of battery-joint
    # planning; (gurobi_no_battery - alns) = exact vs heuristic search quality.
    disable_battery: bool = False

def _node_name(index: int) -> str:
    return f"n{int(index)}"

def _node_index(name: str, origin_node_by_name: Mapping[str, int]) -> int:
    if name.startswith("n"):
        return int(name[1:])
    if name in origin_node_by_name:
        return int(origin_node_by_name[name])
    raise KeyError(f"Cannot map node name to index: {name}")

def _vehicle_mode_for_exact(vehicle: Vehicle) -> str:
    return "UAV" if vehicle.normalized_mode() == "uav" else "ADR"

def _allowed_depot_names(vehicle: Vehicle, n_depots: int) -> List[str]:
    if vehicle.allowed_depots is None:
        return [_node_name(i) for i in range(n_depots)]
    return [_node_name(i) for i in vehicle.allowed_depots]

def _full_battery(vehicle: Vehicle) -> float:
    return float(vehicle.battery_init if vehicle.battery_init is not None else vehicle.battery)

def _make_forced_delivery_or_depot_route(
    vehicle: Vehicle,
    active_requests: Dict[int, Request],
    full_instance: Dict[str, Any],
) -> Optional[List[Leg]]:
    if vehicle.committed_leg is not None:
        return [vehicle.committed_leg]

    onboard = sorted(vehicle.onboard_set())
    if onboard:
        route: List[Leg] = []
        state = vehicle

        for req_id in onboard:
            req = active_requests.get(req_id)
            if req is None:
                continue

            if state.current_node >= int(full_instance["n_depots"]):
                depot_needed = False
                if not _is_arc_feasible(full_instance, state.normalized_mode(), state.current_node, req.delivery_node):
                    depot_needed = True
                else:
                    dm = _get_distance_matrix(full_instance, state.normalized_mode())
                    distance = float(dm[state.current_node, req.delivery_node])
                    energy = _edge_energy_kj(
                        mode=state.normalized_mode(),
                        distance=distance,
                        payload=float(state.load),
                        wind_vec=_wind_vec(full_instance),
                        speed=_cruise_speed(state.normalized_mode()),
                    )
                    depot_needed = state.battery - energy < _battery_threshold(state)

                if depot_needed:
                    depot = _nearest_feasible_depot(state.current_node, state, full_instance)
                    if depot is not None:
                        leg, state, ok = _simulate_leg(state, None, "depot", depot, full_instance)
                        if ok and leg is not None:
                            route.append(leg)

            leg, state, ok = _simulate_leg(state, req, "delivery", req.delivery_node, full_instance)
            if ok and leg is not None:
                route.append(leg)

        return route

    if float(vehicle.battery) < _battery_threshold(vehicle):
        depot = _nearest_feasible_depot(vehicle.current_node, vehicle, full_instance)
        if depot is None:
            return []
        leg, _, ok = _simulate_leg(vehicle, None, "depot", depot, full_instance)
        return [leg] if ok and leg is not None else []

    return None

def _build_exact_data_from_residual(
    *,
    module: Any,
    residual: Dict[str, Any],
    config: ExactRollingConfig,
    available_vehicle_ids: Sequence[int],
) -> Tuple[Any, Dict[str, int], Dict[str, str], Dict[str, int], Dict[str, Tuple[int, str]]]:
    full_instance = residual["full_instance"]
    active_requests: Dict[int, Request] = residual["active_requests"]
    vehicles: Dict[int, Vehicle] = residual["vehicles"]

    n_depots = int(full_instance["n_depots"])
    expected_depots = config.n_depots_uav + config.n_depots_adr
    if expected_depots != n_depots:
        raise ValueError(f"Depot split mismatch: config={expected_depots}, instance={n_depots}")

    waiting_requests = [
        req for req in active_requests.values()
        if req.status == "waiting_for_pickup"
    ]

    exact_requests = []
    node_to_action: Dict[str, Tuple[int, str]] = {}

    for req in waiting_requests:
        pickup_name = _node_name(req.pickup_node)
        delivery_name = _node_name(req.delivery_node)
        exact_requests.append(
            module.Request(
                id=f"r{req.req_id}",
                pickup=pickup_name,
                delivery=delivery_name,
                demand=float(req.demand),
                pickup_target=float(req.t_pickup),
                pickup_latest=float(req.t_pickup + config.latest_pickup_slack),
                delivery_target=float(req.t_delivery),
                delivery_latest=float(req.t_delivery + config.latest_delivery_slack),
                market="shared",
                owner_mode=None,
            )
        )
        node_to_action[pickup_name] = (req.req_id, "pickup")
        node_to_action[delivery_name] = (req.req_id, "delivery")

    exact_vehicles = []
    exact_to_real_vid: Dict[str, int] = {}
    origin_name_by_exact: Dict[str, str] = {}
    origin_node_by_name: Dict[str, int] = {}

    for real_vid in available_vehicle_ids:
        vehicle = vehicles[real_vid]
        exact_vid = f"v{real_vid}"
        exact_to_real_vid[exact_vid] = real_vid

        origin_name = f"origin_v{real_vid}"
        origin_name_by_exact[exact_vid] = origin_name
        origin_node_by_name[origin_name] = int(vehicle.current_node)

        battery_max = max(1e-6, float(vehicle.battery))
        full_battery = max(1e-6, _full_battery(vehicle))
        battery_min = (0.25 if vehicle.normalized_mode() == "uav" else 0.20) * full_battery
        battery_min = min(battery_min, battery_max)

        if config.disable_battery:
            # Pure PDPTW ablation: infinite battery, no depot visits for recharge.
            # Routes are still executed against real battery physics so the gap
            # (gurobi - gurobi_no_battery) measures the value of battery-joint planning.
            battery_max = 1e12
            battery_min = 0.0

        if config.use_discrete_pickup_speed:
            if vehicle.normalized_mode() == "uav":
                speeds = np.linspace(module.V_UAV_MIN_PICKUP, module.V_UAV_MAX, config.pickup_speed_grid_size)
            else:
                speeds = np.linspace(module.V_ADR_MIN_PICKUP, module.V_ADR_MAX, config.pickup_speed_grid_size)
            speed_keys = tuple(f"{float(speed):.6g}" for speed in speeds)
        else:
            speed_keys = ()

        exact_vehicles.append(
            module.Vehicle(
                id=exact_vid,
                mode=_vehicle_mode_for_exact(vehicle),
                capacity=float(vehicle.capacity),
                battery_max=battery_max,
                battery_min=battery_min,
                depot_visit_limit=0 if config.disable_battery else int(config.depot_visit_limit),
                speed_levels=speed_keys,
            )
        )

    real_depot_names = [_node_name(i) for i in range(n_depots)]
    origin_names = list(origin_node_by_name)
    physical_depots = real_depot_names + origin_names

    all_depots = real_depot_names + origin_names
    uav_real_depots = [_node_name(i) for i in range(config.n_depots_uav)]
    adr_real_depots = [_node_name(i) for i in range(config.n_depots_uav, n_depots)]

    uav_origin_names = [
        origin_name_by_exact[f"v{vid}"]
        for vid in available_vehicle_ids
        if vehicles[vid].normalized_mode() == "uav"
    ]
    adr_origin_names = [
        origin_name_by_exact[f"v{vid}"]
        for vid in available_vehicle_ids
        if vehicles[vid].normalized_mode() == "adr"
    ]

    if config.depot_sharing:
        uav_depots = all_depots
        adr_depots = all_depots
    else:
        uav_depots = uav_real_depots + uav_origin_names
        adr_depots = adr_real_depots + adr_origin_names

    physical_customer_names = sorted(set(node_to_action))

    physical_names = physical_depots + physical_customer_names
    travel_time: Dict[Tuple[str, str, str], float] = {}
    energy: Dict[Tuple[str, str, str], float] = {}
    pickup_time_by_speed: Dict[Tuple[str, str, str, str], float] = {}
    pickup_energy_by_speed: Dict[Tuple[str, str, str, str], float] = {}

    for exact_vehicle in exact_vehicles:
        real_vid = exact_to_real_vid[exact_vehicle.id]
        vehicle = vehicles[real_vid]
        mode = vehicle.normalized_mode()

        for source_name in physical_names:
            for target_name in physical_names:
                if source_name == target_name:
                    continue

                source_idx = _node_index(source_name, origin_node_by_name)
                target_idx = _node_index(target_name, origin_node_by_name)

                if source_idx < 0 or target_idx < 0:
                    continue
                if source_idx < n_depots and target_idx < n_depots:
                    continue
                if not _is_arc_feasible(full_instance, mode, source_idx, target_idx):
                    continue

                dm = _get_distance_matrix(full_instance, mode)
                distance = float(dm[source_idx, target_idx])

                target_is_real_depot = target_idx < n_depots and target_name.startswith("n")
                speed = _depot_speed(mode) if source_idx >= n_depots and target_is_real_depot else _cruise_speed(mode)

                req_payload = 0.0
                if target_name in node_to_action:
                    req_id, action = node_to_action[target_name]
                    req = active_requests[req_id]
                    req_payload = float(req.demand) if action in {"pickup", "delivery"} else 0.0

                travel_time[exact_vehicle.id, source_name, target_name] = distance / max(speed, 1e-9)
                energy[exact_vehicle.id, source_name, target_name] = (
                    0.0 if config.disable_battery else
                    _edge_energy_kj(
                        mode=mode,
                        distance=distance,
                        payload=req_payload,
                        wind_vec=_wind_vec(full_instance),
                        speed=speed,
                    )
                )

                if config.use_discrete_pickup_speed and target_name in node_to_action and node_to_action[target_name][1] == "pickup":
                    if mode == "uav":
                        speed_values = np.linspace(module.V_UAV_MIN_PICKUP, module.V_UAV_MAX, config.pickup_speed_grid_size)
                    else:
                        speed_values = np.linspace(module.V_ADR_MIN_PICKUP, module.V_ADR_MAX, config.pickup_speed_grid_size)

                    for speed_value in speed_values:
                        speed_key = f"{float(speed_value):.6g}"
                        pickup_time_by_speed[exact_vehicle.id, source_name, target_name, speed_key] = (
                            distance / max(float(speed_value), 1e-9)
                        )
                        pickup_energy_by_speed[exact_vehicle.id, source_name, target_name, speed_key] = (
                            0.0 if config.disable_battery else
                            _edge_energy_kj(
                                mode=mode,
                                distance=distance,
                                payload=req_payload,
                                wind_vec=_wind_vec(full_instance),
                                speed=float(speed_value),
                            )
                        )

    data = module.CECPDPTWData(
        requests=exact_requests,
        vehicles=exact_vehicles,
        physical_depots=physical_depots,
        uav_depots=uav_depots,
        adr_depots=adr_depots,
        travel_time=travel_time,
        energy=energy,
        pickup_time_by_speed=pickup_time_by_speed,
        pickup_energy_by_speed=pickup_energy_by_speed,
        depot_sharing=config.depot_sharing,
        alpha_uav=0.60,
        alpha_adr=0.10,
        alpha_early_pickup=0.02,
        alpha_late_pickup=0.10,
        alpha_late_delivery=0.15,
        lambda_battery=0.0,
        epsilon_shared=1.0,
        epsilon_to_adr=1.0,
        epsilon_to_uav=1.0,
        uav_customer_service_time=2.0,
        adr_customer_service_time=0.0,
        uav_recharge_duration=10.0,
        adr_recharge_duration=20.0,
        big_m_time=100_000.0,
        big_m_load=100_000.0,
        big_m_battery=1_000_000.0,
    )

    return data, exact_to_real_vid, origin_name_by_exact, origin_node_by_name, node_to_action

def _patch_gurobi_origin_starts(built: Any, origin_name_by_exact: Mapping[str, str]) -> None:
    gp = importlib.import_module("gurobipy")
    model = built.model

    for exact_vid, own_origin in origin_name_by_exact.items():
        for copy_node in built.depot_copies_by_vehicle.get(exact_vid, []):
            meta = built.depot_meta[copy_node]

            outgoing = gp.quicksum(
                built.x[k, i, j]
                for k, i, j in built.arcs
                if k == exact_vid and i == copy_node
            )
            incoming = gp.quicksum(
                built.x[k, i, j]
                for k, i, j in built.arcs
                if k == exact_vid and j == copy_node
            )

            if meta.role == "start":
                if meta.physical_depot_id == own_origin:
                    model.addConstr(outgoing == built.use_vehicle[exact_vid], name=f"force_origin_start[{exact_vid}]")
                else:
                    model.addConstr(outgoing == 0, name=f"block_non_origin_start[{exact_vid},{copy_node}]")

            if meta.physical_depot_id.startswith("origin_") and meta.role != "start":
                model.addConstr(incoming == 0, name=f"block_origin_revisit[{exact_vid},{copy_node}]")

    model.update()

def _patch_ortools_origin_starts(built: Any, origin_name_by_exact: Mapping[str, str]) -> None:
    solver = built.solver

    for exact_vid, own_origin in origin_name_by_exact.items():
        for copy_node in built.depot_copies_by_vehicle.get(exact_vid, []):
            meta = built.depot_meta[copy_node]

            outgoing = solver.Sum([
                built.x[k, i, j]
                for k, i, j in built.arcs
                if k == exact_vid and i == copy_node
            ])
            incoming = solver.Sum([
                built.x[k, i, j]
                for k, i, j in built.arcs
                if k == exact_vid and j == copy_node
            ])

            if meta.role == "start":
                if meta.physical_depot_id == own_origin:
                    solver.Add(outgoing == built.use_vehicle[exact_vid])
                else:
                    solver.Add(outgoing == 0)

            if meta.physical_depot_id.startswith("origin_") and meta.role != "start":
                solver.Add(incoming == 0)

def _extract_exact_routes_to_legs(
    *,
    built: Any,
    residual: Dict[str, Any],
    exact_to_real_vid: Mapping[str, int],
    origin_node_by_name: Mapping[str, int],
    node_to_action: Mapping[str, Tuple[int, str]],
    value_fn,
) -> Dict[int, List[Leg]]:
    active_requests: Dict[int, Request] = residual["active_requests"]
    vehicles: Dict[int, Vehicle] = residual["vehicles"]
    full_instance = residual["full_instance"]

    successors: Dict[Tuple[str, str], str] = {}

    for k, i, j in built.arcs:
        if value_fn(built.x[k, i, j]) >= 0.5:
            successors[k, i] = j

    routes: Dict[int, List[Leg]] = {real_vid: [] for real_vid in exact_to_real_vid.values()}

    for exact_vid, real_vid in exact_to_real_vid.items():
        start_node = None

        for copy_node in built.depot_copies_by_vehicle.get(exact_vid, []):
            meta = built.depot_meta[copy_node]
            if meta.role == "start" and (exact_vid, copy_node) in successors:
                start_node = copy_node
                break

        if start_node is None:
            continue

        sequence = [start_node]
        seen = set()
        current = start_node

        while current is not None and current not in seen:
            seen.add(current)
            nxt = successors.get((exact_vid, current))
            if nxt is None:
                break
            sequence.append(nxt)
            current = nxt

        state = vehicles[real_vid]
        materialized: List[Leg] = []

        for next_node in sequence[1:]:
            if next_node in built.depot_meta:
                physical_name = built.depot_meta[next_node].physical_depot_id
            else:
                physical_name = next_node

            if physical_name.startswith("origin_"):
                continue

            to_index = _node_index(physical_name, origin_node_by_name)

            if physical_name in node_to_action:
                req_id, leg_type = node_to_action[physical_name]
                req = active_requests.get(req_id)
                if req is None:
                    break
                leg, state, ok = _simulate_leg(state, req, leg_type, to_index, full_instance)
            else:
                leg, state, ok = _simulate_leg(state, None, "depot", to_index, full_instance)

            if not ok or leg is None:
                break

            materialized.append(leg)

        routes[real_vid] = materialized

    return routes

class GurobiRollingHorizonSolver:
    def __init__(self, config: ExactRollingConfig):
        self.config = config

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        module = importlib.import_module("ce_cpdptw_merged_gurobi")

        vehicles: Dict[int, Vehicle] = residual["vehicles"]
        active_requests: Dict[int, Request] = residual["active_requests"]
        full_instance = residual["full_instance"]

        routes: Dict[int, List[Leg]] = {vid: [] for vid in vehicles}
        available_vehicle_ids: List[int] = []

        for vid, vehicle in vehicles.items():
            forced = _make_forced_delivery_or_depot_route(vehicle, active_requests, full_instance)
            if forced is not None:
                routes[vid] = forced
            else:
                available_vehicle_ids.append(vid)

        waiting_exists = any(req.status == "waiting_for_pickup" for req in active_requests.values())
        if not waiting_exists or not available_vehicle_ids:
            return routes

        data, exact_to_real_vid, origin_name_by_exact, origin_node_by_name, node_to_action = _build_exact_data_from_residual(
            module=module,
            residual=residual,
            config=self.config,
            available_vehicle_ids=available_vehicle_ids,
        )

        if not data.requests or not data.vehicles:
            return routes

        built = module.build_ce_cpdptw_model(
            data,
            use_adaptive_speed=self.config.use_discrete_pickup_speed,
            tighten_battery_equalities=True,
            model_name="CE_CPDPTW_RH_Gurobi",
        )
        _patch_gurobi_origin_starts(built, origin_name_by_exact)

        model = built.model
        model.Params.LogToConsole = 1 if self.config.log_to_console else 0
        model.Params.TimeLimit = float(self.config.time_limit_seconds)
        model.Params.MIPGap = float(self.config.mip_gap)
        if self.config.n_threads > 0:
            model.Params.Threads = self.config.n_threads
        model.optimize()

        status = getattr(module, "GRB").OPTIMAL
        feasible_statuses = {
            module.GRB.OPTIMAL,
            module.GRB.TIME_LIMIT,
            module.GRB.SUBOPTIMAL,
        }
        if model.Status not in feasible_statuses or model.SolCount == 0:
            return routes

        exact_routes = _extract_exact_routes_to_legs(
            built=built,
            residual=residual,
            exact_to_real_vid=exact_to_real_vid,
            origin_node_by_name=origin_node_by_name,
            node_to_action=node_to_action,
            value_fn=lambda var: float(var.X),
        )
        routes.update(exact_routes)
        return routes

class ORToolsRollingHorizonSolver:
    def __init__(self, config: ExactRollingConfig, solver_id: str = "SCIP"):
        self.config = config
        self.solver_id = solver_id

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        module = importlib.import_module("ce_cpdptw_ortools")
        pywraplp = importlib.import_module("ortools.linear_solver.pywraplp")

        vehicles: Dict[int, Vehicle] = residual["vehicles"]
        active_requests: Dict[int, Request] = residual["active_requests"]
        full_instance = residual["full_instance"]

        routes: Dict[int, List[Leg]] = {vid: [] for vid in vehicles}
        available_vehicle_ids: List[int] = []

        for vid, vehicle in vehicles.items():
            forced = _make_forced_delivery_or_depot_route(vehicle, active_requests, full_instance)
            if forced is not None:
                routes[vid] = forced
            else:
                available_vehicle_ids.append(vid)

        waiting_exists = any(req.status == "waiting_for_pickup" for req in active_requests.values())
        if not waiting_exists or not available_vehicle_ids:
            return routes

        data, exact_to_real_vid, origin_name_by_exact, origin_node_by_name, node_to_action = _build_exact_data_from_residual(
            module=module,
            residual=residual,
            config=self.config,
            available_vehicle_ids=available_vehicle_ids,
        )

        if not data.requests or not data.vehicles:
            return routes

        built = module.build_ce_cpdptw_ortools_model(
            data,
            solver_id=self.solver_id,
            use_adaptive_speed=self.config.use_discrete_pickup_speed,
            tighten_battery_equalities=True,
        )
        _patch_ortools_origin_starts(built, origin_name_by_exact)

        built.solver.SetTimeLimit(int(1000 * self.config.time_limit_seconds))
        if self.config.log_to_console:
            built.solver.EnableOutput()
        else:
            built.solver.SuppressOutput()

        params = pywraplp.MPSolverParameters()
        params.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, float(self.config.mip_gap))
        built.status = built.solver.Solve(params)

        feasible_statuses = {
            pywraplp.Solver.OPTIMAL,
            pywraplp.Solver.FEASIBLE,
        }
        if built.status not in feasible_statuses:
            return routes

        exact_routes = _extract_exact_routes_to_legs(
            built=built,
            residual=residual,
            exact_to_real_vid=exact_to_real_vid,
            origin_node_by_name=origin_node_by_name,
            node_to_action=node_to_action,
            value_fn=lambda var: float(var.solution_value()),
        )
        routes.update(exact_routes)
        return routes

def _infer_shift_minutes(full_instance: Dict[str, Any], extra_minutes: float = 120.0) -> float:
    tw = _to_numpy(full_instance["time_window"]).reshape(-1).astype(float)
    return float(np.nanmax(tw) + extra_minutes)

def solve_static_instance_with_gurobi_rolling(
    full_instance: Dict[str, Any],
    config: ExactRollingConfig,
) -> List[Dict[str, Any]]:
    solver = GurobiRollingHorizonSolver(config)
    fleet = build_initial_fleet_from_instance(
        full_instance,
        SyntheticALNSConfig(
            n_uav=config.n_uav,
            n_adr=config.n_adr,
            n_depots_uav=config.n_depots_uav,
            n_depots_adr=config.n_depots_adr,
            depot_sharing=config.depot_sharing,
            delta_minutes=config.delta_minutes,
            shift_minutes=config.shift_minutes,
        ),
    )
    arrival_stream = build_dynamic_arrival_stream_from_instance(full_instance)
    dispatcher = RollingHorizonDispatcher(
        solver=solver,
        delta_minutes=config.delta_minutes,
        shift_minutes=config.shift_minutes or _infer_shift_minutes(full_instance),
    )
    return dispatcher.run_shift(arrival_stream, fleet, full_instance)

def solve_static_instance_with_gurobi_no_battery_rolling(
    full_instance: Dict[str, Any],
    config: ExactRollingConfig,
) -> List[Dict[str, Any]]:
    """Gurobi ablation: MILP solved as pure PDPTW (all energy values = 0, no depot copies).

    Routes are executed against real battery physics, so service rate drops where
    battery is the binding constraint.  The gap vs solve_static_instance_with_gurobi_rolling
    isolates the value of battery-joint planning; the gap vs ALNS isolates search quality.
    """
    no_bat_config = ExactRollingConfig(
        n_uav=config.n_uav,
        n_adr=config.n_adr,
        n_depots_uav=config.n_depots_uav,
        n_depots_adr=config.n_depots_adr,
        depot_sharing=config.depot_sharing,
        depot_visit_limit=config.depot_visit_limit,
        delta_minutes=config.delta_minutes,
        shift_minutes=config.shift_minutes,
        latest_pickup_slack=config.latest_pickup_slack,
        latest_delivery_slack=config.latest_delivery_slack,
        use_discrete_pickup_speed=config.use_discrete_pickup_speed,
        pickup_speed_grid_size=config.pickup_speed_grid_size,
        time_limit_seconds=config.time_limit_seconds,
        mip_gap=config.mip_gap,
        log_to_console=config.log_to_console,
        n_threads=config.n_threads,
        disable_battery=True,
    )
    return solve_static_instance_with_gurobi_rolling(full_instance, no_bat_config)


def solve_static_instance_with_ortools_rolling(
    full_instance: Dict[str, Any],
    config: ExactRollingConfig,
    *,
    solver_id: str = "SCIP",
) -> List[Dict[str, Any]]:
    solver = ORToolsRollingHorizonSolver(config, solver_id=solver_id)
    fleet = build_initial_fleet_from_instance(
        full_instance,
        SyntheticALNSConfig(
            n_uav=config.n_uav,
            n_adr=config.n_adr,
            n_depots_uav=config.n_depots_uav,
            n_depots_adr=config.n_depots_adr,
            depot_sharing=config.depot_sharing,
            delta_minutes=config.delta_minutes,
            shift_minutes=config.shift_minutes,
        ),
    )
    arrival_stream = build_dynamic_arrival_stream_from_instance(full_instance)
    dispatcher = RollingHorizonDispatcher(
        solver=solver,
        delta_minutes=config.delta_minutes,
        shift_minutes=config.shift_minutes or _infer_shift_minutes(full_instance),
    )
    return dispatcher.run_shift(arrival_stream, fleet, full_instance)

def solve_static_instance_with_all_rolling_baselines(
    full_instance: Dict[str, Any],
    *,
    n_uav: int,
    n_adr: int,
    n_depots_uav: int,
    n_depots_adr: int,
    depot_sharing: bool = True,
    delta_minutes: float = 5.0,
    shift_minutes: Optional[float] = None,
    exact_time_limit_seconds: float = 3600.0,
    exact_mip_gap: float = 0.01,
    ortools_solver_id: str = "SCIP",
    alns_seed: int = 42,
    alns_time_budget_small_s: float = 3.0,
    alns_time_budget_large_s: float = 8.0,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    alns_config = SyntheticALNSConfig(
        n_uav=n_uav,
        n_adr=n_adr,
        n_depots_uav=n_depots_uav,
        n_depots_adr=n_depots_adr,
        depot_sharing=depot_sharing,
        delta_minutes=delta_minutes,
        shift_minutes=shift_minutes,
    )

    exact_config = ExactRollingConfig(
        n_uav=n_uav,
        n_adr=n_adr,
        n_depots_uav=n_depots_uav,
        n_depots_adr=n_depots_adr,
        depot_sharing=depot_sharing,
        delta_minutes=delta_minutes,
        shift_minutes=shift_minutes,
        time_limit_seconds=exact_time_limit_seconds,
        mip_gap=exact_mip_gap,
        log_to_console=False,
    )

    try:
        alns_log = solve_static_instance_with_alns(
            full_instance,
            alns_config,
            solver=ALNSBaseline(
                seed=alns_seed,
                time_budget_small_s=alns_time_budget_small_s,
                time_budget_large_s=alns_time_budget_large_s,
            ),
        )
        report["alns"] = {
            "available": True,
            "summary": summarize_episode_log(alns_log),
        }
    except Exception as exc:
        report["alns"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        gurobi_log = solve_static_instance_with_gurobi_rolling(full_instance, exact_config)
        report["gurobi"] = {
            "available": True,
            "summary": summarize_episode_log(gurobi_log),
        }
    except Exception as exc:
        report["gurobi"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        ortools_log = solve_static_instance_with_ortools_rolling(
            full_instance,
            exact_config,
            solver_id=ortools_solver_id,
        )
        report["ortools"] = {
            "available": True,
            "summary": summarize_episode_log(ortools_log),
        }
    except Exception as exc:
        report["ortools"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    return report

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def demo_compare_toy() -> None:
    full_instance = build_toy_full_instance()
    report = solve_static_instance_with_all_rolling_baselines(
        full_instance,
        n_uav=1,
        n_adr=1,
        n_depots_uav=1,
        n_depots_adr=1,
        depot_sharing=False,
        delta_minutes=2.0,
        shift_minutes=80.0,
        exact_time_limit_seconds=30.0,
        exact_mip_gap=0.01,
        ortools_solver_id="SCIP",
        alns_time_budget_small_s=1.0,
        alns_time_budget_large_s=2.0,
    )
    print(json.dumps(_json_safe(report), indent=2))

if __name__ == "__main__":
    demo_compare_toy()
