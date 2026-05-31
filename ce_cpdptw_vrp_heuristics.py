"""
Pure-Python VRP heuristics for CE-PDPTW rolling-horizon dispatch.

Three solvers implementing  solve(residual) → Dict[int, List[Leg]]:
  ClarkeWrightSolver    — pairwise savings merging (Clarke & Wright 1964)
  RegretInsertionSolver — regret-2 insertion    (Ropke & Pisinger 2006)
  VNSSolver             — Variable Neighborhood Search (Mladenović & Hansen 1997)

All three are model-conformant:
  - Heterogeneous UAV / ADR fleet with energy and capacity constraints
  - Battery recharge legs injected via _simulate_leg / _nearest_feasible_depot
  - Same rolling-horizon RollingHorizonDispatcher interface as the other solvers
  - Both online (dynamic t_arrival) and offline (all-at-t0) wrappers supplied
"""
from __future__ import annotations

import copy
import math
import time
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
    build_static_arrival_stream_from_instance,
    summarize_episode_log,
)
from ce_cpdptw_rolling_baselines import _make_forced_delivery_or_depot_route


# ── shared utilities ──────────────────────────────────────────────────────────

def _needs_recharge(
    vs: Vehicle,
    to_node: int,
    leg_type: str,
    req: Optional[Request],
    full_instance: Dict[str, Any],
) -> bool:
    n_dep = int(full_instance["n_depots"])
    if vs.current_node < n_dep:
        return False
    if not _is_arc_feasible(full_instance, vs.normalized_mode(), vs.current_node, to_node):
        return False
    dm = _get_distance_matrix(full_instance, vs.normalized_mode())
    dist = float(dm[vs.current_node, to_node])
    if leg_type == "pickup" and req is not None:
        spd = _pickup_speed(dist, vs.current_time, req.t_pickup, vs.normalized_mode())
    else:
        spd = _cruise_speed(vs.normalized_mode())
    energy = _edge_energy_kj(
        mode=vs.normalized_mode(), distance=dist, payload=float(vs.load),
        wind_vec=_wind_vec(full_instance), speed=spd,
    )
    return float(vs.battery) - energy < _battery_threshold(vs)


def _tmin(fp: int, tp: int, mode: str, dm: np.ndarray, n_dep: int) -> float:
    """Travel time in minutes including service time at destination."""
    if fp == tp:
        return 0.0
    n = dm.shape[0]
    if fp < 0 or tp < 0 or fp >= n or tp >= n:
        return 1e9
    d = float(dm[fp, tp])
    if not math.isfinite(d) or d >= 1e9:
        return 1e9
    to_dep = tp < n_dep
    spd = _depot_speed(mode) if (to_dep and fp >= n_dep) else _cruise_speed(mode)
    svc = _service_time(mode) if not to_dep else 0.0
    return d / max(spd, 1e-9) + svc


def _route_cost(
    stops: List[Tuple[int, str]],
    start: int,
    mode: str,
    dm: np.ndarray,
    n_dep: int,
    req_list: List[Request],
) -> float:
    pos, cost = start, 0.0
    for ri, lt in stops:
        to = req_list[ri].pickup_node if lt == "pickup" else req_list[ri].delivery_node
        cost += _tmin(pos, to, mode, dm, n_dep)
        pos = to
    return cost


def _check_cap(
    stops: List[Tuple[int, str]],
    capacity: float,
    init_load: float,
    req_list: List[Request],
) -> bool:
    load = init_load
    for ri, lt in stops:
        load += req_list[ri].demand if lt == "pickup" else -req_list[ri].demand
        if load < -1e-6 or load > capacity + 1e-6:
            return False
    return True


def _check_prec(stops: List[Tuple[int, str]]) -> bool:
    seen: set = set()
    for ri, lt in stops:
        if lt == "pickup":
            seen.add(ri)
        elif ri not in seen:
            return False
    return True


def _insertion_delta(
    stops: List[Tuple[int, str]],
    p: int,   # insert pickup before position p  (0 = first)
    d: int,   # insert delivery before position d (d >= p+1)
    ri: int,
    req_list: List[Request],
    start: int,
    mode: str,
    dm: np.ndarray,
    n_dep: int,
) -> float:
    """Cost increase from inserting request ri's pickup at index p and delivery at index d."""
    def phys(idx: int, after_insert: bool = False) -> int:
        """Physical node of stop at index idx (in the stops list, before/after pickup insert)."""
        return req_list[stops[idx][0]].pickup_node if stops[idx][1] == "pickup" \
               else req_list[stops[idx][0]].delivery_node

    pick_phys = req_list[ri].pickup_node
    dliv_phys = req_list[ri].delivery_node
    n = len(stops)

    def node_at(idx: int) -> int:
        if idx < 0:
            return start
        if idx >= n:
            return -1  # no successor
        r, lt = stops[idx]
        return req_list[r].pickup_node if lt == "pickup" else req_list[r].delivery_node

    # Cost before pickup insertion
    a = node_at(p - 1)
    b = node_at(p)
    cost_before_p = _tmin(a, b, mode, dm, n_dep) if b >= 0 else 0.0
    cost_after_p = (_tmin(a, pick_phys, mode, dm, n_dep) +
                    (_tmin(pick_phys, b, mode, dm, n_dep) if b >= 0 else 0.0))
    delta_p = cost_after_p - cost_before_p

    # After pickup inserted, effective index of delivery position is d (in the new sequence)
    # stops has pickup inserted at p, so stop at new index d is old stop at d-1
    def node_at_after(idx: int) -> int:
        if idx < 0:
            return start
        adj = idx if idx <= p else idx - 1  # map new index back to original stops
        if adj >= n:
            return -1
        r, lt = stops[adj]
        return req_list[r].pickup_node if lt == "pickup" else req_list[r].delivery_node

    # But for d > p, pickup is now at position p in the expanded list
    # The node before d in expanded list:
    def exp_node(idx: int) -> int:
        """Node at position idx in sequence [stops[0..p-1], pick, stops[p..], dliv, stops[d..]]."""
        if idx == p:
            return pick_phys
        if idx < p:
            return node_at(idx)
        # idx > p: maps to stops[idx-1]
        return node_at(idx - 1)

    c_before_d = node_at_after(d - 1)
    c_after_d  = node_at_after(d)

    cost_before_d = _tmin(c_before_d, c_after_d, mode, dm, n_dep) if c_after_d >= 0 else 0.0
    cost_after_d = (_tmin(c_before_d, dliv_phys, mode, dm, n_dep) +
                    (_tmin(dliv_phys, c_after_d, mode, dm, n_dep) if c_after_d >= 0 else 0.0))
    delta_d = cost_after_d - cost_before_d

    return delta_p + delta_d


def _best_insertion(
    ri: int,
    stops: List[Tuple[int, str]],
    veh: Vehicle,
    req_list: List[Request],
    dm: np.ndarray,
    n_dep: int,
) -> Tuple[float, int, int]:
    """Best (pickup_pos, delivery_pos) insertion for request ri into stops."""
    mode = veh.normalized_mode()
    start = int(veh.current_node)
    n = len(stops)
    best_cost, best_p, best_d = 1e18, 0, 1
    for p in range(n + 1):
        for d in range(p + 1, n + 2):
            candidate = list(stops)
            candidate.insert(p, (ri, "pickup"))
            candidate.insert(d, (ri, "delivery"))
            if not _check_cap(candidate, veh.capacity, veh.load, req_list):
                continue
            if not _check_prec(candidate):
                continue
            delta = _insertion_delta(stops, p, d, ri, req_list, start, mode, dm, n_dep)
            if delta < best_cost:
                best_cost, best_p, best_d = delta, p, d
    return best_cost, best_p, best_d


def _materialize(
    stops: List[Tuple[int, str]],
    veh_state: Vehicle,
    req_list: List[Request],
    full_instance: Dict[str, Any],
) -> Tuple[List[Leg], Vehicle]:
    legs: List[Leg] = []
    for ri, lt in stops:
        req = req_list[ri]
        to_node = req.pickup_node if lt == "pickup" else req.delivery_node
        if _needs_recharge(veh_state, to_node, lt, req, full_instance):
            depot = _nearest_feasible_depot(veh_state.current_node, veh_state, full_instance)
            if depot is not None:
                leg, veh_state, ok = _simulate_leg(veh_state, None, "depot", depot, full_instance)
                if ok and leg is not None:
                    legs.append(leg)
        leg, veh_state, ok = _simulate_leg(veh_state, req, lt, to_node, full_instance)
        if ok and leg is not None:
            legs.append(leg)
        else:
            break
    return legs, veh_state


def _setup(
    residual: Dict[str, Any],
) -> Tuple[
    Dict[int, List[Leg]],   # routes (pre-filled forced)
    List[int],              # available_vehicle_ids
    List[Vehicle],          # veh_list
    List[Request],          # req_list (waiting_for_pickup only)
    Any,                    # full_instance
    float,                  # current_time
    int,                    # n_depots
]:
    vehicles       = residual["vehicles"]
    active_req     = residual["active_requests"]
    full_instance  = residual["full_instance"]
    current_time   = float(residual.get("current_time", 0.0))

    routes: Dict[int, List[Leg]] = {vid: [] for vid in vehicles}
    available_vids: List[int] = []

    for vid, veh in vehicles.items():
        forced = _make_forced_delivery_or_depot_route(veh, active_req, full_instance)
        if forced is not None:
            routes[vid] = forced
        else:
            available_vids.append(vid)

    req_list = [r for r in active_req.values() if r.status == "waiting_for_pickup"]
    veh_list = [vehicles[vid] for vid in available_vids]
    n_depots = int(full_instance["n_depots"])

    return routes, available_vids, veh_list, req_list, full_instance, current_time, n_depots


# ── 1. Clarke-Wright Savings ──────────────────────────────────────────────────

class ClarkeWrightSolver:
    """
    Savings-based constructive heuristic (Clarke & Wright 1964) adapted for PDPTW.

    Savings: s(i,j,k) = cost(depot→j) − cost(delivery_i→pickup_j)
    where "depot" is vehicle k's current position.  Routes are chained by
    decreasing savings, respecting capacity and pickup-before-delivery precedence.
    Unassigned requests fall back to cheapest-insertion.
    """

    def __init__(self, latest_pickup_slack: float = 30.0, latest_delivery_slack: float = 30.0) -> None:
        self.latest_pickup_slack = latest_pickup_slack
        self.latest_delivery_slack = latest_delivery_slack

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        routes, available_vids, veh_list, req_list, full_instance, current_time, n_depots = _setup(residual)
        vehicles = residual["vehicles"]

        if not req_list or not available_vids:
            return routes

        n_veh = len(available_vids)
        n_req = len(req_list)

        dms = {
            "uav": _get_distance_matrix(full_instance, "uav"),
            "adr": _get_distance_matrix(full_instance, "adr"),
        }

        # Per-vehicle route stops
        veh_stops: List[List[Tuple[int, str]]] = [[] for _ in range(n_veh)]
        assigned = set()

        # Step 1: Savings matrix — for each (i, j, k), savings from chaining j after i on veh k
        savings = []
        for k, veh in enumerate(veh_list):
            mode = veh.normalized_mode()
            dm = dms[mode]
            start = int(veh.current_node)
            for i in range(n_req):
                for j in range(n_req):
                    if i == j:
                        continue
                    ri, rj = req_list[i], req_list[j]
                    # cost of going from start to j's pickup (if j were served alone)
                    cost_to_j = _tmin(start, rj.pickup_node, mode, dm, n_depots)
                    # cost of going from i's delivery directly to j's pickup (chaining)
                    cost_chain = _tmin(ri.delivery_node, rj.pickup_node, mode, dm, n_depots)
                    s = cost_to_j - cost_chain
                    savings.append((s, k, i, j))

        savings.sort(key=lambda x: -x[0])

        # Step 2: Greedily chain routes by savings
        # route_tail[k] = last request index in vehicle k's route (for chaining check)
        route_tail: Dict[int, int] = {}  # k -> last req index

        for s, k, i, j in savings:
            if s <= 0:
                break
            if i in assigned or j in assigned:
                continue
            # Check that i is at the tail of vehicle k's route (or route is empty)
            if veh_stops[k] and route_tail.get(k) != i:
                continue
            # Check we can append (pickup_i, delivery_i, pickup_j, delivery_j)
            candidate = list(veh_stops[k])
            if not veh_stops[k]:
                candidate.extend([(i, "pickup"), (i, "delivery")])
            candidate.extend([(j, "pickup"), (j, "delivery")])
            if not _check_cap(candidate, veh_list[k].capacity, veh_list[k].load, req_list):
                continue
            if not _check_prec(candidate):
                continue
            if not veh_stops[k]:
                veh_stops[k].extend([(i, "pickup"), (i, "delivery")])
                assigned.add(i)
            veh_stops[k].extend([(j, "pickup"), (j, "delivery")])
            assigned.add(j)
            route_tail[k] = j

        # Step 3: Assign remaining unassigned via cheapest insertion
        for i in range(n_req):
            if i in assigned:
                continue
            best_k, best_p, best_d, best_delta = 0, 0, 1, 1e18
            for k, veh in enumerate(veh_list):
                dm = dms[veh.normalized_mode()]
                delta, p, d = _best_insertion(i, veh_stops[k], veh, req_list, dm, n_depots)
                if delta < best_delta:
                    best_delta, best_k, best_p, best_d = delta, k, p, d
            veh_stops[best_k].insert(best_p, (i, "pickup"))
            veh_stops[best_k].insert(best_d, (i, "delivery"))
            assigned.add(i)

        # Step 4: Materialize
        for k, vid in enumerate(available_vids):
            vs = copy.deepcopy(vehicles[vid])
            if vs.battery_init is None:
                vs.battery_init = float(vs.battery)
            legs, _ = _materialize(veh_stops[k], vs, req_list, full_instance)
            routes[vid] = legs

        return routes


# ── 2. Regret-2 Insertion ─────────────────────────────────────────────────────

class RegretInsertionSolver:
    """
    Regret-2 insertion heuristic (Ropke & Pisinger 2006).

    At each step, for every unassigned request i:
      regret(i) = 2nd-best insertion cost − best insertion cost
    The request with the highest regret is inserted at its best position.
    High regret = large opportunity cost of not inserting now.
    """

    def __init__(self, latest_pickup_slack: float = 30.0, latest_delivery_slack: float = 30.0) -> None:
        self.latest_pickup_slack = latest_pickup_slack
        self.latest_delivery_slack = latest_delivery_slack

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        routes, available_vids, veh_list, req_list, full_instance, current_time, n_depots = _setup(residual)
        vehicles = residual["vehicles"]

        if not req_list or not available_vids:
            return routes

        n_veh = len(available_vids)
        dms = {
            "uav": _get_distance_matrix(full_instance, "uav"),
            "adr": _get_distance_matrix(full_instance, "adr"),
        }

        veh_stops: List[List[Tuple[int, str]]] = [[] for _ in range(n_veh)]
        unassigned = set(range(len(req_list)))

        while unassigned:
            best_choice = None  # (regret, ri, best_k, best_p, best_d)

            for ri in unassigned:
                # Find best and second-best insertion across all vehicles
                insertions = []
                for k, veh in enumerate(veh_list):
                    dm = dms[veh.normalized_mode()]
                    delta, p, d = _best_insertion(ri, veh_stops[k], veh, req_list, dm, n_depots)
                    if delta < 1e17:
                        insertions.append((delta, k, p, d))
                insertions.sort()

                if not insertions:
                    # No feasible insertion — skip for now (will be inserted last)
                    continue

                c1 = insertions[0][0]
                c2 = insertions[1][0] if len(insertions) >= 2 else c1 + 1e9
                regret = c2 - c1

                if best_choice is None or regret > best_choice[0]:
                    best_choice = (regret, ri, insertions[0][1], insertions[0][2], insertions[0][3])

            if best_choice is None:
                # Infeasible for all: force into least-loaded vehicle at tail
                ri = next(iter(unassigned))
                loads = [
                    sum(req_list[s[0]].demand for s in veh_stops[k] if s[1] == "pickup")
                    for k in range(n_veh)
                ]
                best_k = int(np.argmin(loads))
                veh_stops[best_k].extend([(ri, "pickup"), (ri, "delivery")])
                unassigned.discard(ri)
            else:
                _, ri, k, p, d = best_choice
                veh_stops[k].insert(p, (ri, "pickup"))
                veh_stops[k].insert(d, (ri, "delivery"))
                unassigned.discard(ri)

        for k, vid in enumerate(available_vids):
            vs = copy.deepcopy(vehicles[vid])
            if vs.battery_init is None:
                vs.battery_init = float(vs.battery)
            legs, _ = _materialize(veh_stops[k], vs, req_list, full_instance)
            routes[vid] = legs

        return routes


# ── 3. Variable Neighborhood Search ──────────────────────────────────────────

class VNSSolver:
    """
    Variable Neighborhood Search (Mladenović & Hansen 1997) for PDPTW.

    Initial solution: greedy cheapest insertion.
    Neighborhoods (applied in order, reset to N1 on improvement):
      N1 — Relocate: move one (pickup, delivery) pair to a cheaper position
           in any vehicle (same or different).
      N2 — Swap: exchange two (pickup, delivery) pairs between any two vehicles.
    Terminates when time budget exhausted or no neighborhood improves.
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

    # ── internal helpers ──────────────────────────────────────────────────

    def _total_cost(self, veh_stops, veh_list, req_list, dms, n_depots):
        return sum(
            _route_cost(
                veh_stops[k],
                int(veh_list[k].current_node),
                veh_list[k].normalized_mode(),
                dms[veh_list[k].normalized_mode()],
                n_depots,
                req_list,
            )
            for k in range(len(veh_list))
        )

    def _feasible(self, stops, veh, req_list):
        return _check_cap(stops, veh.capacity, veh.load, req_list) and _check_prec(stops)

    def _greedy_init(self, req_list, veh_list, dms, n_depots):
        n_veh = len(veh_list)
        veh_stops: List[List[Tuple[int, str]]] = [[] for _ in range(n_veh)]
        for ri in range(len(req_list)):
            best_k, best_p, best_d, best_delta = 0, 0, 1, 1e18
            for k, veh in enumerate(veh_list):
                dm = dms[veh.normalized_mode()]
                delta, p, d = _best_insertion(ri, veh_stops[k], veh, req_list, dm, n_depots)
                if delta < best_delta:
                    best_delta, best_k, best_p, best_d = delta, k, p, d
            veh_stops[best_k].insert(best_p, (ri, "pickup"))
            veh_stops[best_k].insert(best_d, (ri, "delivery"))
        return veh_stops

    def _relocate(self, veh_stops, veh_list, req_list, dms, n_depots):
        """N1: try moving each request to a better position anywhere."""
        n_veh = len(veh_list)
        improved = False
        current_cost = self._total_cost(veh_stops, veh_list, req_list, dms, n_depots)

        for src_k in range(n_veh):
            req_indices = sorted({ri for ri, _ in veh_stops[src_k]})
            for ri in req_indices:
                # Remove ri from src_k
                new_src = [(r, lt) for r, lt in veh_stops[src_k] if r != ri]
                for dst_k in range(n_veh):
                    dm = dms[veh_list[dst_k].normalized_mode()]
                    base = list(new_src) if dst_k == src_k else list(veh_stops[dst_k])
                    delta, p, d = _best_insertion(ri, base, veh_list[dst_k], req_list, dm, n_depots)
                    if delta >= 1e17:
                        continue
                    candidate_dst = list(base)
                    candidate_dst.insert(p, (ri, "pickup"))
                    candidate_dst.insert(d, (ri, "delivery"))
                    if not self._feasible(candidate_dst, veh_list[dst_k], req_list):
                        continue

                    # Evaluate new total cost
                    new_stops = [list(s) for s in veh_stops]
                    new_stops[src_k] = new_src
                    new_stops[dst_k] = candidate_dst
                    new_cost = self._total_cost(new_stops, veh_list, req_list, dms, n_depots)
                    if new_cost < current_cost - 1e-6:
                        for k in range(n_veh):
                            veh_stops[k] = new_stops[k]
                        current_cost = new_cost
                        improved = True
                        break  # restart outer loop
                if improved:
                    break
            if improved:
                break

        return improved

    def _swap(self, veh_stops, veh_list, req_list, dms, n_depots):
        """N2: try swapping any two requests between routes."""
        n_veh = len(veh_list)
        improved = False
        current_cost = self._total_cost(veh_stops, veh_list, req_list, dms, n_depots)

        req_by_veh = [{ri for ri, _ in veh_stops[k]} for k in range(n_veh)]

        for k1 in range(n_veh):
            for ri in sorted(req_by_veh[k1]):
                for k2 in range(n_veh):
                    for rj in sorted(req_by_veh[k2]):
                        if ri == rj:
                            continue
                        # Swap ri (from k1) and rj (from k2)
                        s1 = [(r, lt) for r, lt in veh_stops[k1] if r != ri]
                        s2 = [(r, lt) for r, lt in veh_stops[k2] if r != rj]

                        # Insert rj into s1 and ri into s2
                        dm1 = dms[veh_list[k1].normalized_mode()]
                        dm2 = dms[veh_list[k2].normalized_mode()]
                        d1, p1, dd1 = _best_insertion(rj, s1, veh_list[k1], req_list, dm1, n_depots)
                        d2, p2, dd2 = _best_insertion(ri, s2, veh_list[k2], req_list, dm2, n_depots)

                        if d1 >= 1e17 or d2 >= 1e17:
                            continue

                        new_s1 = list(s1)
                        new_s1.insert(p1, (rj, "pickup"))
                        new_s1.insert(dd1, (rj, "delivery"))
                        new_s2 = list(s2)
                        new_s2.insert(p2, (ri, "pickup"))
                        new_s2.insert(dd2, (ri, "delivery"))

                        if not self._feasible(new_s1, veh_list[k1], req_list):
                            continue
                        if not self._feasible(new_s2, veh_list[k2], req_list):
                            continue

                        new_stops = [list(s) for s in veh_stops]
                        new_stops[k1] = new_s1
                        new_stops[k2] = new_s2
                        new_cost = self._total_cost(new_stops, veh_list, req_list, dms, n_depots)

                        if new_cost < current_cost - 1e-6:
                            for k in range(n_veh):
                                veh_stops[k] = new_stops[k]
                            current_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        return improved

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        routes, available_vids, veh_list, req_list, full_instance, current_time, n_depots = _setup(residual)
        vehicles = residual["vehicles"]

        if not req_list or not available_vids:
            return routes

        dms = {
            "uav": _get_distance_matrix(full_instance, "uav"),
            "adr": _get_distance_matrix(full_instance, "adr"),
        }

        # Initial solution: greedy insertion
        veh_stops = self._greedy_init(req_list, veh_list, dms, n_depots)

        # VNS main loop
        deadline = time.perf_counter() + self.time_limit_seconds
        k_max = 2  # number of neighborhoods
        k = 0
        while k < k_max and time.perf_counter() < deadline:
            if k == 0:
                improved = self._relocate(veh_stops, veh_list, req_list, dms, n_depots)
            else:
                improved = self._swap(veh_stops, veh_list, req_list, dms, n_depots)
            k = 0 if improved else k + 1

        # Materialize
        for k_idx, vid in enumerate(available_vids):
            vs = copy.deepcopy(vehicles[vid])
            if vs.battery_init is None:
                vs.battery_init = float(vs.battery)
            legs, _ = _materialize(veh_stops[k_idx], vs, req_list, full_instance)
            routes[vid] = legs

        return routes


# ── wrapper functions (online + offline for each solver) ──────────────────────

def _make_wrapper(solver_cls, solver_kwargs=None):
    """Return online and offline solve functions for the given solver class."""

    def _solve_rolling(
        full_instance: Dict[str, Any],
        *,
        n_uav: int,
        n_adr: int,
        n_depots_uav: int,
        n_depots_adr: int,
        depot_sharing: bool = True,
        delta_minutes: float = 5.0,
        shift_minutes: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        config = SyntheticALNSConfig(
            n_uav=n_uav, n_adr=n_adr,
            n_depots_uav=n_depots_uav, n_depots_adr=n_depots_adr,
            depot_sharing=depot_sharing,
            delta_minutes=delta_minutes,
            shift_minutes=shift_minutes,
        )
        fleet = build_initial_fleet_from_instance(full_instance, config)
        arrival_stream = build_dynamic_arrival_stream_from_instance(full_instance)
        if shift_minutes is None:
            tw = _to_numpy(full_instance["time_window"]).reshape(-1).astype(float)
            shift_minutes = float(np.nanmax(tw) + 120.0)
        kw = (solver_kwargs or {}).copy()
        kw.update({k: v for k, v in kwargs.items()
                   if k in ("time_limit_seconds", "latest_pickup_slack", "latest_delivery_slack")})
        solver = solver_cls(**kw)
        dispatcher = RollingHorizonDispatcher(
            solver=solver, delta_minutes=delta_minutes, shift_minutes=shift_minutes,
        )
        return dispatcher.run_shift(arrival_stream, fleet, full_instance)

    def _solve_offline(
        full_instance: Dict[str, Any],
        *,
        n_uav: int,
        n_adr: int,
        n_depots_uav: int,
        n_depots_adr: int,
        depot_sharing: bool = True,
        delta_minutes: float = 5.0,
        shift_minutes: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        config = SyntheticALNSConfig(
            n_uav=n_uav, n_adr=n_adr,
            n_depots_uav=n_depots_uav, n_depots_adr=n_depots_adr,
            depot_sharing=depot_sharing,
            delta_minutes=delta_minutes,
            shift_minutes=shift_minutes,
        )
        fleet = build_initial_fleet_from_instance(full_instance, config)
        # Static stream: all requests at t=0 (oracle full information)
        arrival_stream = build_static_arrival_stream_from_instance(full_instance)
        if shift_minutes is None:
            tw = _to_numpy(full_instance["time_window"]).reshape(-1).astype(float)
            shift_minutes = float(np.nanmax(tw) + 120.0)
        kw = (solver_kwargs or {}).copy()
        kw.update({k: v for k, v in kwargs.items()
                   if k in ("time_limit_seconds", "latest_pickup_slack", "latest_delivery_slack")})
        solver = solver_cls(**kw)
        dispatcher = RollingHorizonDispatcher(
            solver=solver, delta_minutes=delta_minutes, shift_minutes=shift_minutes,
        )
        return dispatcher.run_shift(arrival_stream, fleet, full_instance)

    return _solve_rolling, _solve_offline


(
    solve_static_instance_with_cw_rolling,
    solve_static_instance_with_cw_offline,
) = _make_wrapper(ClarkeWrightSolver)

(
    solve_static_instance_with_regret_rolling,
    solve_static_instance_with_regret_offline,
) = _make_wrapper(RegretInsertionSolver)

(
    solve_static_instance_with_vns_rolling,
    solve_static_instance_with_vns_offline,
) = _make_wrapper(VNSSolver)
