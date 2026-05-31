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

_TIME_SCALE = 100
_BIG_M_INT = int(1e8)


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


class CPSATVRPRollingHorizonSolver:
    """
    Rolling-horizon VRP solver using OR-Tools CP-SAT (ortools.sat.python.cp_model).

    Replaces the pywrapcp RoutingModel which segfaults on CC cluster due to its
    C++ thread pool calling Python transit callbacks without holding the GIL.
    CP-SAT uses a SAT/LP backend with no Python callbacks — purely C++ internally.

    Model: one add_circuit per vehicle over optional pickup/delivery nodes.
    Self-loop literals handle unassigned nodes.  Time propagation via
    only_enforce_if conditional constraints.  Objective: minimize total transit.
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
        from ortools.sat.python import cp_model

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

        n_veh = len(available_vehicle_ids)
        n_req = len(waiting_requests)
        n_depots = int(full_instance["n_depots"])

        veh_list = [vehicles[vid] for vid in available_vehicle_ids]
        req_list = list(waiting_requests)

        # CP-SAT node layout (per-vehicle circuit):
        #   PICK(i)  = 2*i          — pickup of request i
        #   DLIV(i)  = 2*i + 1      — delivery of request i
        #   DEPOT(k) = 2*n_req + k  — start/end depot for vehicle k
        # Total nodes N = 2*n_req + n_veh
        N = 2 * n_req + n_veh

        def PICK(i: int) -> int: return 2 * i
        def DLIV(i: int) -> int: return 2 * i + 1
        def DEPOT(k: int) -> int: return 2 * n_req + k

        # Physical grid node for each CP-SAT node
        phys: List[int] = [0] * N
        for i, req in enumerate(req_list):
            phys[PICK(i)] = req.pickup_node
            phys[DLIV(i)] = req.delivery_node
        for k, veh in enumerate(veh_list):
            phys[DEPOT(k)] = int(veh.current_node)

        # Transit times (scaled integers, precomputed — no C++ calls inside solver)
        TSCALE = 100
        HORIZON = int(1000 * TSCALE)
        dm_uav = _get_distance_matrix(full_instance, "uav").tolist()
        dm_adr = _get_distance_matrix(full_instance, "adr").tolist()

        def _trav(fp: int, tp: int, mode: str) -> int:
            if fp == tp:
                return 0
            dm = dm_uav if mode == "uav" else dm_adr
            n_p = len(dm)
            if fp < 0 or tp < 0 or fp >= n_p or tp >= n_p:
                return HORIZON
            d = float(dm[fp][tp])
            if not math.isfinite(d) or d >= 1e9:
                return HORIZON
            to_dep = tp < n_depots
            from_cust = fp >= n_depots
            spd = _depot_speed(mode) if (to_dep and from_cust) else _cruise_speed(mode)
            tm = d / max(spd, 1e-9)
            svc = _service_time(mode) if not to_dep else 0.0
            return int((tm + svc) * TSCALE)

        modes = [v.normalized_mode() for v in veh_list]

        # tran[k][n][m] = integer travel time for vehicle k from CP node n to m.
        # Pure distance/speed transit — no battery adjustment here.
        # Battery is path-dependent (state at arc k depends on all prior arcs) so
        # approximating recharge overhead with starting battery causes false infeasibility:
        # inflated transit >> delivery time window → CP-SAT skips all requests (0% service).
        # Energy is handled reactively during route materialization (_needs_recharge /
        # _simulate_leg) which evaluates the actual battery state at each stop.
        tran: List[List[List[int]]] = [
            [[_trav(phys[n], phys[m], modes[k]) for m in range(N)] for n in range(N)]
            for k in range(n_veh)
        ]

        # Time windows (scaled integers, relative to current_time)
        tw_lo: List[int] = [0] * N
        tw_hi: List[int] = [HORIZON] * N
        for i, req in enumerate(req_list):
            ep = max(0.0, req.t_arrival - current_time)
            lp = max(req.t_pickup - current_time + self.latest_pickup_slack, ep + 1.0)
            ld = max(req.t_delivery - current_time + self.latest_delivery_slack, 0.0)
            tw_lo[PICK(i)] = int(ep * TSCALE)
            tw_hi[PICK(i)] = int(lp * TSCALE)
            tw_hi[DLIV(i)] = int(ld * TSCALE)

        # ── Build CP-SAT model ──────────────────────────────────────────────
        model = cp_model.CpModel()

        # Self-loop literals: loop[k][n] is True when vehicle k does NOT visit node n.
        # Used in add_circuit as the self-loop arc literal.
        loop = [
            [model.new_bool_var(f'lp_{k}_{n}') for n in range(N)]
            for k in range(n_veh)
        ]

        # Each vehicle cannot visit other vehicles' depots.
        # Its own depot is conditionally forced into the circuit:
        #   loop[k][DEPOT(k)] <= loop[k][PICK(i)]  for all i
        # → depot self-loop is forbidden whenever any pickup is visited (forces depot
        #   into the circuit as start/end).
        # → depot CAN self-loop when NO requests are served (all pickups skipped),
        #   keeping add_circuit feasible when every request is time-window-infeasible.
        # Unconditionally forcing loop[k][DEPOT(k)]==0 caused INFEASIBLE whenever all
        # requests were skipped — same symptom as the old add_exactly_one bug.
        for k in range(n_veh):
            for k2 in range(n_veh):
                if k2 != k:
                    model.add(loop[k][DEPOT(k2)] == 1)
            for i in range(n_req):
                model.add(loop[k][DEPOT(k)] <= loop[k][PICK(i)])

        # Each request served by at most one vehicle; pickup and delivery same vehicle.
        # at_most_one (not exactly_one) so that infeasible requests don't make the
        # whole model INFEASIBLE — the solver will skip them and pay SERVE_PENALTY.
        for i in range(n_req):
            model.add_at_most_one([loop[k][PICK(i)].negated() for k in range(n_veh)])
            for k in range(n_veh):
                # pickup visited ↔ delivery visited (same vehicle constraint)
                model.add(loop[k][PICK(i)] == loop[k][DLIV(i)])

        # Arc variables: arc[k][(n,m)] is True when vehicle k travels directly n→m.
        arc: List[Dict[Tuple[int, int], Any]] = [
            {(n, m): model.new_bool_var(f'a_{k}_{n}_{m}')
             for n in range(N) for m in range(N) if n != m}
            for k in range(n_veh)
        ]

        # Circuit constraint: each vehicle's arcs + self-loops form a valid circuit.
        for k in range(n_veh):
            circuit_arcs = [(n, n, loop[k][n]) for n in range(N)]
            circuit_arcs += [(n, m, arc[k][(n, m)]) for (n, m) in arc[k]]
            model.add_circuit(circuit_arcs)

        # Time variables: t[k][n] = arrival time of vehicle k at CP node n.
        t = [
            [model.new_int_var(0, HORIZON, f't_{k}_{n}') for n in range(N)]
            for k in range(n_veh)
        ]

        # Vehicle departs its own depot at time 0.
        for k in range(n_veh):
            model.add(t[k][DEPOT(k)] == 0)

        # Time propagation: if arc k n→m is used, t[k][m] ≥ t[k][n] + transit.
        for k in range(n_veh):
            for (n, m), a_var in arc[k].items():
                tr = tran[k][n][m]
                if tr > 0:
                    model.add(t[k][m] >= t[k][n] + tr).only_enforce_if(a_var)

        # Time windows and precedence (only enforced when node is visited).
        for i in range(n_req):
            for k in range(n_veh):
                visits = loop[k][PICK(i)].negated()
                model.add(t[k][PICK(i)] >= tw_lo[PICK(i)]).only_enforce_if(visits)
                model.add(t[k][PICK(i)] <= tw_hi[PICK(i)]).only_enforce_if(visits)
                model.add(t[k][DLIV(i)] <= tw_hi[DLIV(i)]).only_enforce_if(visits)
                model.add(t[k][DLIV(i)] >= t[k][PICK(i)]).only_enforce_if(visits)

        # Objective: minimize total transit + penalty for unserved requests.
        # SERVE_PENALTY >> max possible transit per request, so serving is always
        # preferred when feasible. loop[k][PICK(i)]=1 means vehicle k skips pickup i;
        # marginal cost of not serving request i = SERVE_PENALTY (one extra loop=1
        # compared to when it is served by one vehicle).
        SERVE_PENALTY = 10 * HORIZON
        arc_vars: List[Any] = []
        coeffs: List[int] = []
        for k in range(n_veh):
            for (n, m), a_var in arc[k].items():
                c = tran[k][n][m]
                if 0 < c < HORIZON:
                    arc_vars.append(a_var)
                    coeffs.append(c)
        for i in range(n_req):
            for k in range(n_veh):
                arc_vars.append(loop[k][PICK(i)])
                coeffs.append(SERVE_PENALTY)
        if arc_vars:
            model.minimize(cp_model.LinearExpr.weighted_sum(arc_vars, coeffs))

        # Solve
        cp_solver = cp_model.CpSolver()
        cp_solver.parameters.max_time_in_seconds = self.time_limit_seconds
        cp_solver.parameters.log_search_progress = False
        status = cp_solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return routes

        # ── Materialise routes as Leg objects ───────────────────────────────
        for k, vid in enumerate(available_vehicle_ids):
            veh_state = copy.deepcopy(vehicles[vid])
            if veh_state.battery_init is None:
                veh_state.battery_init = float(veh_state.battery)

            # Follow arc chain from DEPOT(k) until returning to DEPOT(k).
            cur = DEPOT(k)
            visit_order: List[int] = []
            for _ in range(N + 1):
                nxt = None
                for m in range(N):
                    if m != cur and (cur, m) in arc[k] and cp_solver.boolean_value(arc[k][(cur, m)]):
                        nxt = m
                        break
                if nxt is None or nxt == DEPOT(k):
                    break
                visit_order.append(nxt)
                cur = nxt

            legs: List[Leg] = []
            for cp_node in visit_order:
                if cp_node >= 2 * n_req:
                    continue  # other vehicle's depot node — skip
                i = cp_node // 2
                is_pickup = (cp_node % 2 == 0)
                req = req_list[i]
                leg_type = "pickup" if is_pickup else "delivery"
                to_node = req.pickup_node if is_pickup else req.delivery_node

                if _needs_recharge(veh_state, to_node, leg_type, req, full_instance):
                    depot = _nearest_feasible_depot(veh_state.current_node, veh_state, full_instance)
                    if depot is not None:
                        leg, veh_state, ok = _simulate_leg(veh_state, None, "depot", depot, full_instance)
                        if ok and leg is not None:
                            legs.append(leg)
                        elif not ok:
                            break  # recharge leg failed — vehicle cannot proceed

                leg, veh_state, ok = _simulate_leg(veh_state, req, leg_type, to_node, full_instance)
                if ok and leg is not None:
                    legs.append(leg)
                else:
                    break

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

    solver = CPSATVRPRollingHorizonSolver(
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
