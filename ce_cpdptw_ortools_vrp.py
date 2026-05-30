from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ce_cpdptw_alns import (
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
    build_dynamic_arrival_stream_from_instance,
    build_initial_fleet_from_instance,
    summarize_episode_log,
)
from ce_cpdptw_rolling_baselines import _make_forced_delivery_or_depot_route

# Integer time scale: 1 OR-Tools unit = 1/TIME_SCALE minutes
_TIME_SCALE = 100
_BIG_M_INT = int(1e8)


def _travel_int(
    from_phys: int,
    to_phys: int,
    mode: str,
    n_depots: int,
    dm_uav: np.ndarray,
    dm_adr: np.ndarray,
) -> int:
    if from_phys == to_phys:
        return 0
    dm = dm_uav if mode == "uav" else dm_adr
    n_phys = dm.shape[0]
    if from_phys < 0 or to_phys < 0 or from_phys >= n_phys or to_phys >= n_phys:
        return _BIG_M_INT
    dist = float(dm[from_phys, to_phys])
    if not math.isfinite(dist) or dist >= 1e9:
        return _BIG_M_INT
    to_is_depot = to_phys < n_depots
    from_is_customer = from_phys >= n_depots
    speed = _depot_speed(mode) if (to_is_depot and from_is_customer) else _cruise_speed(mode)
    travel_min = dist / max(speed, 1e-9)
    svc = _service_time(mode) if not to_is_depot else 0.0
    return int((travel_min + svc) * _TIME_SCALE)


def _needs_recharge(
    vehicle_state: Vehicle,
    to_node: int,
    leg_type: str,
    req: Optional[Request],
    full_instance: Dict[str, Any],
) -> bool:
    n_depots = int(full_instance["n_depots"])
    if vehicle_state.current_node < n_depots:
        return False
    if not _is_arc_feasible(full_instance, vehicle_state.normalized_mode(), vehicle_state.current_node, to_node):
        return False
    dm = _get_distance_matrix(full_instance, vehicle_state.normalized_mode())
    distance = float(dm[vehicle_state.current_node, to_node])
    if leg_type == "pickup" and req is not None:
        speed = _pickup_speed(distance, vehicle_state.current_time, req.t_pickup, vehicle_state.normalized_mode())
    else:
        speed = _cruise_speed(vehicle_state.normalized_mode())
    energy_needed = _edge_energy_kj(
        mode=vehicle_state.normalized_mode(),
        distance=distance,
        payload=float(vehicle_state.load),
        wind_vec=_wind_vec(full_instance),
        speed=speed,
    )
    return float(vehicle_state.battery) - energy_needed < _battery_threshold(vehicle_state)


class ORToolsVRPRollingHorizonSolver:
    """
    Rolling-horizon solver using OR-Tools pywrapcp.RoutingModel (VRP routing API,
    not MILP/SCIP). Each re-plan runs in ~5-10 seconds using GLS metaheuristic.

    Heterogeneous fleet: UAV and ADR vehicles get separate per-vehicle transit
    callbacks via AddDimensionWithVehicleTransits so travel times match their
    respective speeds.
    """

    def __init__(
        self,
        time_limit_seconds: float = 8.0,
        latest_pickup_slack: float = 30.0,
        latest_delivery_slack: float = 30.0,
    ) -> None:
        self.time_limit_seconds = time_limit_seconds
        self.latest_pickup_slack = latest_pickup_slack
        self.latest_delivery_slack = latest_delivery_slack

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        vehicles: Dict[int, Vehicle] = residual["vehicles"]
        active_requests: Dict[int, Request] = residual["active_requests"]
        full_instance = residual["full_instance"]
        current_time: float = float(residual.get("current_time", 0.0))

        routes: Dict[int, List[Leg]] = {vid: [] for vid in vehicles}
        available_vehicle_ids: List[int] = []

        for vid, vehicle in vehicles.items():
            forced = _make_forced_delivery_or_depot_route(vehicle, active_requests, full_instance)
            if forced is not None:
                routes[vid] = forced
            else:
                available_vehicle_ids.append(vid)

        waiting_requests = [
            req for req in active_requests.values()
            if req.status == "waiting_for_pickup"
        ]

        if not waiting_requests or not available_vehicle_ids:
            return routes

        n_vehicles = len(available_vehicle_ids)
        n_requests = len(waiting_requests)
        n_depots = int(full_instance["n_depots"])

        # OR-Tools node layout:
        #   0                                  : dummy end depot
        #   [1 .. n_vehicles]                  : vehicle start positions
        #   [n_vehicles+1 .. n_vehicles+n_req] : pickup nodes
        #   [n_vehicles+n_req+1 .. +2*n_req]   : delivery nodes
        DUMMY_END = 0
        veh_off = 1
        pick_off = 1 + n_vehicles
        dliv_off = 1 + n_vehicles + n_requests
        n_nodes = 1 + n_vehicles + 2 * n_requests

        veh_list = [vehicles[vid] for vid in available_vehicle_ids]
        req_list = list(waiting_requests)

        # Physical grid node for each OR-Tools node
        phys: List[int] = [0] * n_nodes
        phys[DUMMY_END] = 0  # depot 0 as dummy return
        for i, v in enumerate(veh_list):
            phys[veh_off + i] = int(v.current_node)
        for i, r in enumerate(req_list):
            phys[pick_off + i] = int(r.pickup_node)
            phys[dliv_off + i] = int(r.delivery_node)

        dm_uav = _get_distance_matrix(full_instance, "uav")
        dm_adr = _get_distance_matrix(full_instance, "adr")

        starts = list(range(veh_off, veh_off + n_vehicles))
        ends = [DUMMY_END] * n_vehicles

        manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        # Per-vehicle transit callbacks (UAV vs ADR have different speeds)
        transit_indices: List[int] = []
        modes = [v.normalized_mode() for v in veh_list]

        for v_i, mode in enumerate(modes):
            captured_mode = mode

            def _make_cb(m):
                def cb(fi, ti):
                    fp = phys[manager.IndexToNode(fi)]
                    tp = phys[manager.IndexToNode(ti)]
                    return _travel_int(fp, tp, m, n_depots, dm_uav, dm_adr)
                return cb

            cb_idx = routing.RegisterTransitCallback(_make_cb(captured_mode))
            transit_indices.append(cb_idx)
            routing.SetArcCostEvaluatorOfVehicle(cb_idx, v_i)

        # Time dimension with per-vehicle transit
        HORIZON_INT = int(1000 * _TIME_SCALE)
        SLACK_INT = int(60 * _TIME_SCALE)
        routing.AddDimensionWithVehicleTransits(
            transit_indices,
            SLACK_INT,
            HORIZON_INT,
            False,
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        # Capacity dimension
        demand_scaled = [0] * n_nodes
        for i, r in enumerate(req_list):
            demand_scaled[pick_off + i] = int(round(r.demand * 100))
            demand_scaled[dliv_off + i] = -int(round(r.demand * 100))

        def demand_cb(idx):
            return demand_scaled[manager.IndexToNode(idx)]

        demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb_idx,
            0,
            [int(round(v.capacity * 100)) for v in veh_list],
            True,
            "Capacity",
        )

        # Pickup-delivery pairs with same-vehicle and precedence constraints
        solver = routing.solver()
        for i, req in enumerate(req_list):
            pick_idx = manager.NodeToIndex(pick_off + i)
            dliv_idx = manager.NodeToIndex(dliv_off + i)
            routing.AddPickupAndDelivery(pick_idx, dliv_idx)
            solver.Add(routing.VehicleVar(pick_idx) == routing.VehicleVar(dliv_idx))
            solver.Add(time_dim.CumulVar(pick_idx) <= time_dim.CumulVar(dliv_idx))

            # Time windows (relative to current_time)
            earliest_pick = max(0.0, req.t_arrival - current_time)
            latest_pick = max(
                req.t_pickup - current_time + self.latest_pickup_slack,
                earliest_pick + 1.0,
            )
            latest_dliv = max(req.t_delivery - current_time + self.latest_delivery_slack, 0.0)

            time_dim.CumulVar(pick_idx).SetRange(
                int(earliest_pick * _TIME_SCALE),
                int(latest_pick * _TIME_SCALE),
            )
            time_dim.CumulVar(dliv_idx).SetRange(0, int(latest_dliv * _TIME_SCALE))

        # Vehicle start/end time windows
        for v_i in range(n_vehicles):
            time_dim.CumulVar(routing.Start(v_i)).SetRange(0, 0)
            time_dim.CumulVar(routing.End(v_i)).SetRange(0, HORIZON_INT)

        # Search parameters
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        params.time_limit.seconds = int(self.time_limit_seconds)
        params.log_search = False

        solution = routing.SolveWithParameters(params)
        if solution is None:
            return routes

        # Materialise routes as Leg objects via _simulate_leg
        for v_i, vid in enumerate(available_vehicle_ids):
            veh_state = copy.deepcopy(vehicles[vid])
            if veh_state.battery_init is None:
                veh_state.battery_init = float(veh_state.battery)

            legs: List[Leg] = []
            index = routing.Start(v_i)

            while not routing.IsEnd(index):
                next_idx = solution.Value(routing.NextVar(index))
                if routing.IsEnd(next_idx):
                    break
                node = manager.IndexToNode(next_idx)
                if node == DUMMY_END:
                    break

                if pick_off <= node < dliv_off:
                    req = req_list[node - pick_off]
                    leg_type = "pickup"
                    to_node = req.pickup_node
                elif dliv_off <= node < n_nodes:
                    req = req_list[node - dliv_off]
                    leg_type = "delivery"
                    to_node = req.delivery_node
                else:
                    index = next_idx
                    continue

                if _needs_recharge(veh_state, to_node, leg_type, req, full_instance):
                    depot = _nearest_feasible_depot(veh_state.current_node, veh_state, full_instance)
                    if depot is not None:
                        leg, veh_state, ok = _simulate_leg(veh_state, None, "depot", depot, full_instance)
                        if ok and leg is not None:
                            legs.append(leg)

                leg, veh_state, ok = _simulate_leg(veh_state, req, leg_type, to_node, full_instance)
                if ok and leg is not None:
                    legs.append(leg)
                else:
                    break

                index = next_idx

            routes[vid] = legs

        return routes


def solve_static_instance_with_ortools_vrp_rolling(
    full_instance: Dict[str, Any],
    *,
    n_uav: int,
    n_adr: int,
    n_depots_uav: int,
    n_depots_adr: int,
    depot_sharing: bool = True,
    delta_minutes: float = 5.0,
    shift_minutes: Optional[float] = None,
    time_limit_seconds: float = 8.0,
    latest_pickup_slack: float = 30.0,
    latest_delivery_slack: float = 30.0,
) -> List[Dict[str, Any]]:
    config = SyntheticALNSConfig(
        n_uav=n_uav,
        n_adr=n_adr,
        n_depots_uav=n_depots_uav,
        n_depots_adr=n_depots_adr,
        depot_sharing=depot_sharing,
        delta_minutes=delta_minutes,
        shift_minutes=shift_minutes,
    )
    fleet = build_initial_fleet_from_instance(full_instance, config)
    arrival_stream = build_dynamic_arrival_stream_from_instance(full_instance)

    if shift_minutes is None:
        tw = _to_numpy(full_instance["time_window"]).reshape(-1).astype(float)
        shift_minutes = float(np.nanmax(tw) + 120.0)

    solver = ORToolsVRPRollingHorizonSolver(
        time_limit_seconds=time_limit_seconds,
        latest_pickup_slack=latest_pickup_slack,
        latest_delivery_slack=latest_delivery_slack,
    )
    dispatcher = RollingHorizonDispatcher(
        solver=solver,
        delta_minutes=delta_minutes,
        shift_minutes=shift_minutes,
    )
    return dispatcher.run_shift(arrival_stream, fleet, full_instance)
