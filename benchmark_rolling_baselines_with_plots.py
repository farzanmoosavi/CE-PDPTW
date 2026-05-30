from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from ce_cpdptw_alns import (
    ALNSBaseline,
    SyntheticALNSConfig,
    _get_distance_matrix,
    _get_mask_matrix,
    _to_numpy,
    build_toy_full_instance,
    solve_static_instance_with_alns,
)
from ce_cpdptw_rolling_baselines import (
    ExactRollingConfig,
    solve_static_instance_with_gurobi_rolling,
    solve_static_instance_with_ortools_rolling,
)
from ce_cpdptw_ortools_vrp import solve_static_instance_with_ortools_vrp_rolling

SUMMARY_KEYS = [
    "baseline",
    "instance_id",
    "available",
    "status",
    "error",
    "total_revealed",
    "total_delivered",
    "undelivered",
    "delivered_rate",
    "undelivered_rate",
    "total_cost",
    "operating_cost",
    "operating_cost_uav",
    "operating_cost_adr",
    "penalty_cost",
    "battery_penalty_cost",
    "undelivered_penalty",
    "wall_time_s",
    "solver_time_s_total",
    "solver_time_s_mean",
    "hard_constraint_violation_rate",
    "soft_time_window_violation_rate",
    "pickup_early_rate",
    "pickup_late_rate",
    "delivery_late_rate",
    "arc_constraint_violation_rate",
    "capacity_violation_rate",
    "depot_sharing_violation_rate",
    "battery_threshold_violation_epoch_rate",
    "completed_leg_count",
    "infeasible_arc_count",
    "capacity_violation_count",
    "depot_sharing_violation_count",
    "hard_violating_request_count",
    "soft_tw_violating_request_count",
]

def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def get_summary(episode_log: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not episode_log:
        return {}
    if episode_log[-1].get("summary"):
        return dict(episode_log[-1])
    return {}

def iter_epoch_entries(episode_log: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for entry in episode_log:
        if not entry.get("summary"):
            yield entry

def get_capacity_by_vehicle(full_instance: Dict[str, Any]) -> Dict[int, float]:
    capacity = _to_numpy(full_instance["capacity"]).reshape(-1).astype(float)
    return {idx: float(value) for idx, value in enumerate(capacity)}

def is_arc_feasible_from_log(
    full_instance: Dict[str, Any],
    *,
    vehicle_id: int,
    n_uav: int,
    from_node: int,
    to_node: int,
) -> bool:
    mode = "uav" if int(vehicle_id) < n_uav else "adr"

    try:
        dm = _get_distance_matrix(full_instance, mode)
    except Exception:
        return False

    if from_node < 0 or to_node < 0 or from_node >= dm.shape[0] or to_node >= dm.shape[1]:
        return False

    distance = float(dm[from_node, to_node])
    if not math.isfinite(distance) or distance >= 1e9:
        return False

    mask = _get_mask_matrix(full_instance, mode)
    if mask is not None and float(mask[from_node, to_node]) <= 0.5:
        return False

    return True

def is_depot_sharing_violation(
    *,
    vehicle_id: int,
    to_node: int,
    n_uav: int,
    n_depots_uav: int,
    n_depots_adr: int,
    depot_sharing: bool,
) -> bool:
    n_depots = n_depots_uav + n_depots_adr
    if depot_sharing:
        return False
    if to_node >= n_depots:
        return False

    is_uav = int(vehicle_id) < n_uav
    if is_uav:
        return not (0 <= to_node < n_depots_uav)

    return not (n_depots_uav <= to_node < n_depots)

def compute_constraint_metrics(
    episode_log: Sequence[Dict[str, Any]],
    full_instance: Dict[str, Any],
    *,
    n_uav: int,
    n_depots_uav: int,
    n_depots_adr: int,
    depot_sharing: bool,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    summary = get_summary(episode_log)
    request_history = summary.get("request_history", []) or []

    total_revealed = int(summary.get("total_revealed", len(request_history)) or 0)
    total_delivered = int(summary.get("total_delivered", 0) or 0)
    undelivered = int(summary.get("undelivered", max(0, total_revealed - total_delivered)) or 0)

    hard_violating_requests = 0
    soft_tw_violating_requests = 0
    pickup_early = 0
    pickup_late = 0
    delivery_late = 0

    for request in request_history:
        status = request.get("status")
        pickup_actual = request.get("T_pickup_actual")
        delivery_actual = request.get("T_delivery_actual")
        pickup_target = request.get("t_pickup_target")
        delivery_target = request.get("t_delivery_target")

        hard_bad = False
        if status != "delivered":
            hard_bad = True
        if pickup_actual is None or delivery_actual is None:
            hard_bad = True
        elif float(delivery_actual) + tolerance < float(pickup_actual):
            hard_bad = True

        if hard_bad:
            hard_violating_requests += 1

        soft_bad = False
        if pickup_actual is not None and pickup_target is not None:
            if float(pickup_actual) + tolerance < float(pickup_target):
                pickup_early += 1
                soft_bad = True
            if float(pickup_actual) > float(pickup_target) + tolerance:
                pickup_late += 1
                soft_bad = True

        if delivery_actual is not None and delivery_target is not None:
            if float(delivery_actual) > float(delivery_target) + tolerance:
                delivery_late += 1
                soft_bad = True

        if soft_bad:
            soft_tw_violating_requests += 1

    completed_leg_count = 0
    infeasible_arc_count = 0
    capacity_violation_count = 0
    depot_sharing_violation_count = 0
    capacity_by_vehicle = get_capacity_by_vehicle(full_instance)

    epoch_count = 0
    battery_threshold_epoch_count = 0
    solver_times: List[float] = []

    for epoch in iter_epoch_entries(episode_log):
        epoch_count += 1
        solver_times.append(float(epoch.get("solver_time_s", 0.0) or 0.0))

        if float(epoch.get("battery_penalty_step", 0.0) or 0.0) > tolerance:
            battery_threshold_epoch_count += 1

        for leg in epoch.get("completed", []) or []:
            completed_leg_count += 1
            vehicle_id = int(leg.get("vehicle_id", -1))
            from_node = int(leg.get("from_node", -1))
            to_node = int(leg.get("to_node", -1))

            if not is_arc_feasible_from_log(
                full_instance,
                vehicle_id=vehicle_id,
                n_uav=n_uav,
                from_node=from_node,
                to_node=to_node,
            ):
                infeasible_arc_count += 1

            if is_depot_sharing_violation(
                vehicle_id=vehicle_id,
                to_node=to_node,
                n_uav=n_uav,
                n_depots_uav=n_depots_uav,
                n_depots_adr=n_depots_adr,
                depot_sharing=depot_sharing,
            ):
                depot_sharing_violation_count += 1

            load_before = float(leg.get("load_before", 0.0) or 0.0)
            load_after = float(leg.get("load_after", 0.0) or 0.0)
            capacity = float(capacity_by_vehicle.get(vehicle_id, float("inf")))

            if load_before < -tolerance or load_after < -tolerance or load_before > capacity + tolerance or load_after > capacity + tolerance:
                capacity_violation_count += 1

            if float(leg.get("travel_time", 0.0) or 0.0) < -tolerance:
                infeasible_arc_count += 1

            if float(leg.get("energy_used", 0.0) or 0.0) < -tolerance:
                infeasible_arc_count += 1

    return {
        "total_revealed": total_revealed,
        "total_delivered": total_delivered,
        "undelivered": undelivered,
        "delivered_rate": safe_divide(total_delivered, total_revealed),
        "undelivered_rate": safe_divide(undelivered, total_revealed),
        "hard_constraint_violation_rate": safe_divide(hard_violating_requests, total_revealed),
        "soft_time_window_violation_rate": safe_divide(soft_tw_violating_requests, total_revealed),
        "pickup_early_rate": safe_divide(pickup_early, total_revealed),
        "pickup_late_rate": safe_divide(pickup_late, total_revealed),
        "delivery_late_rate": safe_divide(delivery_late, total_revealed),
        "arc_constraint_violation_rate": safe_divide(infeasible_arc_count, completed_leg_count),
        "capacity_violation_rate": safe_divide(capacity_violation_count, completed_leg_count),
        "depot_sharing_violation_rate": safe_divide(depot_sharing_violation_count, completed_leg_count),
        "battery_threshold_violation_epoch_rate": safe_divide(battery_threshold_epoch_count, epoch_count),
        "completed_leg_count": completed_leg_count,
        "infeasible_arc_count": infeasible_arc_count,
        "capacity_violation_count": capacity_violation_count,
        "depot_sharing_violation_count": depot_sharing_violation_count,
        "hard_violating_request_count": hard_violating_requests,
        "soft_tw_violating_request_count": soft_tw_violating_requests,
        "solver_time_s_total": sum(solver_times),
        "solver_time_s_mean": statistics.mean(solver_times) if solver_times else 0.0,
    }

def flatten_episode_result(
    *,
    baseline: str,
    instance_id: int,
    episode_log: Sequence[Dict[str, Any]],
    full_instance: Dict[str, Any],
    wall_time_s: float,
    n_uav: int,
    n_depots_uav: int,
    n_depots_adr: int,
    depot_sharing: bool,
) -> Dict[str, Any]:
    summary = get_summary(episode_log)
    metrics = compute_constraint_metrics(
        episode_log,
        full_instance,
        n_uav=n_uav,
        n_depots_uav=n_depots_uav,
        n_depots_adr=n_depots_adr,
        depot_sharing=depot_sharing,
    )

    row = {
        "baseline": baseline,
        "instance_id": instance_id,
        "available": True,
        "status": "ok",
        "error": "",
        "total_cost": float(summary.get("total_cost", 0.0) or 0.0),
        "operating_cost": float(summary.get("operating_cost", 0.0) or 0.0),
        "operating_cost_uav": float(summary.get("operating_cost_uav", 0.0) or 0.0),
        "operating_cost_adr": float(summary.get("operating_cost_adr", 0.0) or 0.0),
        "penalty_cost": float(summary.get("penalty_cost", 0.0) or 0.0),
        "battery_penalty_cost": float(summary.get("battery_penalty_cost", 0.0) or 0.0),
        "undelivered_penalty": float(summary.get("undelivered_penalty", 0.0) or 0.0),
        "wall_time_s": float(wall_time_s),
    }
    row.update(metrics)
    return row

def failed_row(baseline: str, instance_id: int, error: BaseException, wall_time_s: float) -> Dict[str, Any]:
    row = {key: "" for key in SUMMARY_KEYS}
    row.update({
        "baseline": baseline,
        "instance_id": instance_id,
        "available": False,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "wall_time_s": float(wall_time_s),
    })
    return row

def _arrivals_from_instance(full_instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    n_depots = int(full_instance['n_depots'])
    n_req = int(full_instance['n_req'])

    tw = full_instance['time_window']
    if hasattr(tw, 'cpu'):
        tw_arr = tw.cpu().numpy().reshape(-1)
    else:
        tw_arr = np.array(tw).reshape(-1)

    demand_raw = full_instance['demand']
    if hasattr(demand_raw, 'cpu'):
        demand_arr = demand_raw.cpu().numpy().reshape(-1)
    else:
        demand_arr = np.array(demand_raw).reshape(-1)

    if 't_arrival' in full_instance:
        t_arrival_raw = full_instance['t_arrival']
        if hasattr(t_arrival_raw, 'cpu'):
            t_arrival_arr = t_arrival_raw.cpu().numpy().reshape(-1)
        else:
            t_arrival_arr = np.array(t_arrival_raw).reshape(-1)
    else:
        t_arrival_arr = None

    arrivals = []
    for j in range(n_req):
        pickup_node = n_depots + j
        delivery_node = n_depots + n_req + j
        t_arr = float(t_arrival_arr[j]) if t_arrival_arr is not None else 0.0
        arrivals.append({
            'req_id': j,
            't_arrival': t_arr,
            't_pickup': float(tw_arr[pickup_node]),
            't_delivery': float(tw_arr[delivery_node]),
            'pickup_node': pickup_node,
            'delivery_node': delivery_node,
            'demand': float(abs(demand_arr[pickup_node])),
        })
    return sorted(arrivals, key=lambda x: x['t_arrival'])

def run_one_rl_baseline(
    *,
    rl_solver,
    full_instance: Dict[str, Any],
    instance_id: int,
    n_uav: int,
    n_adr: int,
    n_depots_uav: int,
    n_depots_adr: int,
    depot_sharing: bool,
    delta_minutes: float,
    shift_minutes: Optional[float],
) -> Dict[str, Any]:
    from dispatch_sim import RollingHorizonDispatcher
    from coalition import make_fleet

    start = time.perf_counter()
    try:
        sm = shift_minutes if shift_minutes is not None else 120.0
        fleet = make_fleet(
            n_uav=n_uav, n_adr=n_adr,
            n_depots_uav=n_depots_uav, n_depots_adr=n_depots_adr,
        )
        arrivals = _arrivals_from_instance(full_instance)
        dispatcher = RollingHorizonDispatcher(
            rl_solver, delta_minutes=delta_minutes, shift_minutes=sm,
        )
        episode_log = dispatcher.run_shift(arrivals, fleet, full_instance)
        wall_time = time.perf_counter() - start
        return flatten_episode_result(
            baseline='rl',
            instance_id=instance_id,
            episode_log=episode_log,
            full_instance=full_instance,
            wall_time_s=wall_time,
            n_uav=n_uav,
            n_depots_uav=n_depots_uav,
            n_depots_adr=n_depots_adr,
            depot_sharing=depot_sharing,
        )
    except Exception as exc:
        wall_time = time.perf_counter() - start
        return failed_row('rl', instance_id, exc, wall_time)

def run_one_baseline(
    *,
    baseline: str,
    full_instance: Dict[str, Any],
    instance_id: int,
    n_uav: int,
    n_adr: int,
    n_depots_uav: int,
    n_depots_adr: int,
    depot_sharing: bool,
    delta_minutes: float,
    shift_minutes: Optional[float],
    exact_time_limit_seconds: float,
    exact_mip_gap: float,
    ortools_solver_id: str,
    alns_seed: int,
    alns_small_budget_s: float,
    alns_large_budget_s: float,
    gurobi_threads: int = 0,
) -> Dict[str, Any]:
    start = time.perf_counter()

    try:
        if baseline == "alns":
            config = SyntheticALNSConfig(
                n_uav=n_uav,
                n_adr=n_adr,
                n_depots_uav=n_depots_uav,
                n_depots_adr=n_depots_adr,
                depot_sharing=depot_sharing,
                delta_minutes=delta_minutes,
                shift_minutes=shift_minutes,
            )
            solver = ALNSBaseline(
                seed=alns_seed + instance_id,
                time_budget_small_s=alns_small_budget_s,
                time_budget_large_s=alns_large_budget_s,
            )
            episode_log = solve_static_instance_with_alns(full_instance, config, solver=solver)

        elif baseline == "offline_alns":
            from offline_alns import solve_static_instance_with_offline_alns
            config = SyntheticALNSConfig(
                n_uav=n_uav,
                n_adr=n_adr,
                n_depots_uav=n_depots_uav,
                n_depots_adr=n_depots_adr,
                depot_sharing=depot_sharing,
                delta_minutes=delta_minutes,
                shift_minutes=shift_minutes,
            )
            episode_log = solve_static_instance_with_offline_alns(
                full_instance, config,
                seed=alns_seed + instance_id,
                time_budget_small_s=alns_small_budget_s,
                time_budget_large_s=alns_large_budget_s,
            )

        elif baseline in ("fifo", "greedy"):
            from dispatch_sim import RollingHorizonDispatcher
            from coalition import make_fleet

            if baseline == "fifo":
                from fifo_baseline import FIFOSolver
                inner_solver = FIFOSolver()
            else:
                from greedy_insertion import GreedyInsertion
                inner_solver = GreedyInsertion()

            sm = shift_minutes if shift_minutes is not None else 120.0
            fleet = make_fleet(
                n_uav=n_uav, n_adr=n_adr,
                n_depots_uav=n_depots_uav, n_depots_adr=n_depots_adr,
            )
            arrivals = _arrivals_from_instance(full_instance)
            dispatcher = RollingHorizonDispatcher(
                inner_solver, delta_minutes=delta_minutes, shift_minutes=sm,
            )
            episode_log = dispatcher.run_shift(arrivals, fleet, full_instance)

        elif baseline == "gurobi":
            config = ExactRollingConfig(
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
                n_threads=gurobi_threads,
            )
            episode_log = solve_static_instance_with_gurobi_rolling(full_instance, config)

        elif baseline == "ortools":
            config = ExactRollingConfig(
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
            episode_log = solve_static_instance_with_ortools_rolling(
                full_instance,
                config,
                solver_id=ortools_solver_id,
            )

        elif baseline == "ortools_vrp":
            episode_log = solve_static_instance_with_ortools_vrp_rolling(
                full_instance,
                n_uav=n_uav,
                n_adr=n_adr,
                n_depots_uav=n_depots_uav,
                n_depots_adr=n_depots_adr,
                depot_sharing=depot_sharing,
                delta_minutes=delta_minutes,
                shift_minutes=shift_minutes,
                time_limit_seconds=min(exact_time_limit_seconds, 10.0),
            )

        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        wall_time = time.perf_counter() - start
        return flatten_episode_result(
            baseline=baseline,
            instance_id=instance_id,
            episode_log=episode_log,
            full_instance=full_instance,
            wall_time_s=wall_time,
            n_uav=n_uav,
            n_depots_uav=n_depots_uav,
            n_depots_adr=n_depots_adr,
            depot_sharing=depot_sharing,
        )

    except Exception as exc:
        wall_time = time.perf_counter() - start
        return failed_row(baseline, instance_id, exc, wall_time)

def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    numeric_keys = [
        "total_revealed",
        "total_delivered",
        "undelivered",
        "delivered_rate",
        "undelivered_rate",
        "total_cost",
        "operating_cost",
        "operating_cost_uav",
        "operating_cost_adr",
        "penalty_cost",
        "battery_penalty_cost",
        "undelivered_penalty",
        "wall_time_s",
        "solver_time_s_total",
        "solver_time_s_mean",
        "hard_constraint_violation_rate",
        "soft_time_window_violation_rate",
        "pickup_early_rate",
        "pickup_late_rate",
        "delivery_late_rate",
        "arc_constraint_violation_rate",
        "capacity_violation_rate",
        "depot_sharing_violation_rate",
        "battery_threshold_violation_epoch_rate",
        "completed_leg_count",
        "infeasible_arc_count",
        "capacity_violation_count",
        "depot_sharing_violation_count",
        "hard_violating_request_count",
        "soft_tw_violating_request_count",
    ]

    baselines = sorted({str(row.get("baseline")) for row in rows})
    aggregate: List[Dict[str, Any]] = []

    for baseline in baselines:
        group = [row for row in rows if row.get("baseline") == baseline]
        successful = [row for row in group if row.get("available") is True]

        result: Dict[str, Any] = {
            "baseline": baseline,
            "n_runs": len(group),
            "n_success": len(successful),
            "success_rate": safe_divide(len(successful), len(group)),
        }

        for key in numeric_keys:
            values = []
            for row in successful:
                value = row.get(key)
                if value == "" or value is None:
                    continue
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue

            if values:
                result[f"{key}_mean"] = statistics.mean(values)
                result[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
                result[f"{key}_min"] = min(values)
                result[f"{key}_max"] = max(values)
            else:
                result[f"{key}_mean"] = ""
                result[f"{key}_std"] = ""
                result[f"{key}_min"] = ""
                result[f"{key}_max"] = ""

        aggregate.append(result)

    return aggregate

def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        ordered = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)
        fieldnames = ordered

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")

def make_instance_factory(
    *,
    toy: bool,
    n_req: int,
    n_uav: int,
    n_adr: int,
    n_depots_uav: int,
    n_depots_adr: int,
    base_seed: int,
    demand_range=(1.0, 6.0),
    wind_speed_range=(0.0, 12.0),
    tw_slack_mean=30.0,
    tw_slack_std=5.0,
    tw_slack_clip=(15.0, 60.0),
) -> Callable[[int], Dict[str, Any]]:
    if toy:
        def toy_factory(_: int) -> Dict[str, Any]:
            return build_toy_full_instance()
        return toy_factory

    dataset = importlib.import_module("creat_vrp")

    def dataset_factory(instance_id: int) -> Dict[str, Any]:
        return dataset.create_instance(
            n_req=n_req,
            n_uav=n_uav,
            n_adr=n_adr,
            n_depots_uav=n_depots_uav,
            n_depots_adr=n_depots_adr,
            rng=np.random.default_rng(base_seed + instance_id),
            demand_range=demand_range,
            wind_speed_range=wind_speed_range,
            tw_slack_mean=tw_slack_mean,
            tw_slack_std=tw_slack_std,
            tw_slack_clip=tw_slack_clip,
        )

    return dataset_factory

def _worker_run_instance(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    import numpy as np
    from creat_vrp import create_instance as _create_instance

    instance_id  = task['instance_id']
    baselines    = task['baselines']
    toy          = task['toy']
    gurobi_threads = task['gurobi_threads']

    if toy:
        full_instance = build_toy_full_instance()
    else:
        full_instance = _create_instance(
            n_req=task['n_req'],
            n_uav=task['n_uav'],
            n_adr=task['n_adr'],
            n_depots_uav=task['n_depots_uav'],
            n_depots_adr=task['n_depots_adr'],
            rng=np.random.default_rng(task['base_seed'] + instance_id),
            demand_range=task.get('demand_range', (1.0, 6.0)),
            wind_speed_range=task.get('wind_speed_range', (0.0, 12.0)),
            tw_slack_mean=task.get('tw_slack_mean', 30.0),
            tw_slack_std=task.get('tw_slack_std', 5.0),
            tw_slack_clip=task.get('tw_slack_clip', (15.0, 60.0)),
        )

    rows = []
    for baseline in baselines:
        row = run_one_baseline(
            baseline=baseline,
            full_instance=full_instance,
            instance_id=instance_id,
            n_uav=task['n_uav'],
            n_adr=task['n_adr'],
            n_depots_uav=task['n_depots_uav'],
            n_depots_adr=task['n_depots_adr'],
            depot_sharing=task['depot_sharing'],
            delta_minutes=task['delta_minutes'],
            shift_minutes=task['shift_minutes'],
            exact_time_limit_seconds=task['exact_time_limit'],
            exact_mip_gap=task['exact_mip_gap'],
            ortools_solver_id=task['ortools_solver'],
            alns_seed=task['alns_seed'],
            alns_small_budget_s=task['alns_small_budget'],
            alns_large_budget_s=task['alns_large_budget'],
            gurobi_threads=gurobi_threads,
        )
        rows.append(row)
    return rows

def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    baselines = [item.strip().lower() for item in args.baselines.split(",") if item.strip()]
    valid = {"alns", "offline_alns", "fifo", "greedy", "gurobi", "ortools", "ortools_vrp", "rl"}
    invalid = [baseline for baseline in baselines if baseline not in valid]
    if invalid:
        raise ValueError(f"Unknown baselines: {invalid}")

    cpu_baselines = [b for b in baselines if b != 'rl']
    has_rl = 'rl' in baselines

    rl_solver = None
    if has_rl:
        rl_model_path = getattr(args, 'rl_model', '')
        if not rl_model_path:
            raise ValueError("--rl-model is required when 'rl' is listed in --baselines")
        from main import build_solver
        rl_arch = getattr(args, 'arch', 'hetgat')
        rl_solver = build_solver(rl_model_path, arch=rl_arch)

    _demand_range     = (getattr(args, 'demand_low', 1.0), getattr(args, 'demand_high', 6.0))
    _wind_range       = (getattr(args, 'wind_speed_low', 0.0), getattr(args, 'wind_speed_high', 12.0))
    _tw_slack_mean    = getattr(args, 'tw_slack_mean', 30.0)
    _tw_slack_std     = getattr(args, 'tw_slack_std', 5.0)
    _tw_slack_clip    = (getattr(args, 'tw_slack_clip_low', 15.0), getattr(args, 'tw_slack_clip_high', 60.0))

    instance_factory = make_instance_factory(
        toy=args.toy,
        n_req=args.n_req,
        n_uav=args.n_uav,
        n_adr=args.n_adr,
        n_depots_uav=args.n_depots_uav,
        n_depots_adr=args.n_depots_adr,
        base_seed=args.seed,
        demand_range=_demand_range,
        wind_speed_range=_wind_range,
        tw_slack_mean=_tw_slack_mean,
        tw_slack_std=_tw_slack_std,
        tw_slack_clip=_tw_slack_clip,
    )

    n_workers = getattr(args, 'workers', 1)
    import os as _os
    gurobi_threads = max(1, (_os.cpu_count() or 1) // n_workers) if n_workers > 1 else 0

    per_instance_rows: List[Dict[str, Any]] = []

    if cpu_baselines and n_workers > 1:
        import multiprocessing
        _mp_ctx = multiprocessing.get_context("forkserver")
        from concurrent.futures import ProcessPoolExecutor
        tasks = [
            {
                'instance_id':    instance_id,
                'baselines':      cpu_baselines,
                'toy':            args.toy,
                'n_req':          args.n_req,
                'n_uav':          args.n_uav,
                'n_adr':          args.n_adr,
                'n_depots_uav':   args.n_depots_uav,
                'n_depots_adr':   args.n_depots_adr,
                'base_seed':      args.seed,
                'depot_sharing':  args.depot_sharing,
                'delta_minutes':  args.delta_minutes,
                'shift_minutes':  args.shift_minutes,
                'exact_time_limit': args.exact_time_limit,
                'exact_mip_gap':  args.exact_mip_gap,
                'ortools_solver': args.ortools_solver,
                'alns_seed':      args.seed,
                'alns_small_budget': args.alns_small_budget,
                'alns_large_budget': args.alns_large_budget,
                'gurobi_threads': gurobi_threads,
                'demand_range':    _demand_range,
                'wind_speed_range': _wind_range,
                'tw_slack_mean':   _tw_slack_mean,
                'tw_slack_std':    _tw_slack_std,
                'tw_slack_clip':   _tw_slack_clip,
            }
            for instance_id in range(args.num_instances)
        ]
        print(f'Running {args.num_instances} instances × {len(cpu_baselines)} baselines '
              f'with {n_workers} workers (Gurobi threads/worker={gurobi_threads})')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_mp_ctx) as pool:
            for rows in pool.map(_worker_run_instance, tasks):
                per_instance_rows.extend(rows)
                for row in rows:
                    print(
                        f"[{row['baseline']}] instance={row['instance_id']} "
                        f"available={row.get('available')} "
                        f"cost={row.get('total_cost')} "
                        f"delivered_rate={row.get('delivered_rate')}"
                    )
    else:
        for instance_id in range(args.num_instances):
            full_instance = instance_factory(instance_id)

            for baseline in cpu_baselines:
                row = run_one_baseline(
                    baseline=baseline,
                    full_instance=full_instance,
                    instance_id=instance_id,
                    n_uav=args.n_uav,
                    n_adr=args.n_adr,
                    n_depots_uav=args.n_depots_uav,
                    n_depots_adr=args.n_depots_adr,
                    depot_sharing=args.depot_sharing,
                    delta_minutes=args.delta_minutes,
                    shift_minutes=args.shift_minutes,
                    exact_time_limit_seconds=args.exact_time_limit,
                    exact_mip_gap=args.exact_mip_gap,
                    ortools_solver_id=args.ortools_solver,
                    alns_seed=args.seed,
                    alns_small_budget_s=args.alns_small_budget,
                    alns_large_budget_s=args.alns_large_budget,
                )
                per_instance_rows.append(row)
                print(
                    f"[{baseline}] instance={instance_id} "
                    f"available={row.get('available')} "
                    f"cost={row.get('total_cost')} "
                    f"delivered_rate={row.get('delivered_rate')} "
                    f"hard_cv_rate={row.get('hard_constraint_violation_rate')}"
                )

            if has_rl:
                row = run_one_rl_baseline(
                    rl_solver=rl_solver,
                    full_instance=full_instance,
                    instance_id=instance_id,
                    n_uav=args.n_uav,
                    n_adr=args.n_adr,
                    n_depots_uav=args.n_depots_uav,
                    n_depots_adr=args.n_depots_adr,
                    depot_sharing=args.depot_sharing,
                    delta_minutes=args.delta_minutes,
                    shift_minutes=args.shift_minutes,
                )
                per_instance_rows.append(row)
                print(
                    f"[rl] instance={instance_id} "
                    f"available={row.get('available')} cost={row.get('total_cost')}"
                )

    if has_rl and n_workers > 1:
        print('Running RL baseline sequentially (GPU solver, cannot be forked)')
        for instance_id in range(args.num_instances):
            full_instance = instance_factory(instance_id)
            row = run_one_rl_baseline(
                rl_solver=rl_solver,
                full_instance=full_instance,
                instance_id=instance_id,
                n_uav=args.n_uav,
                n_adr=args.n_adr,
                n_depots_uav=args.n_depots_uav,
                n_depots_adr=args.n_depots_adr,
                depot_sharing=args.depot_sharing,
                delta_minutes=args.delta_minutes,
                shift_minutes=args.shift_minutes,
            )
            per_instance_rows.append(row)
            print(f"[rl] instance={instance_id} cost={row.get('total_cost')}")

    aggregate = aggregate_rows(per_instance_rows)

    output_dir = Path(args.output_dir)
    write_json(output_dir / "per_instance_results.json", per_instance_rows)
    write_csv(output_dir / "per_instance_results.csv", per_instance_rows, fieldnames=SUMMARY_KEYS)
    write_json(output_dir / "aggregate_results.json", aggregate)
    write_csv(output_dir / "aggregate_results.csv", aggregate)

    return {
        "per_instance_rows": per_instance_rows,
        "aggregate": aggregate,
        "output_dir": str(output_dir),
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--toy", action="store_true", help="Use the built-in toy instance instead of dataset.create_instance.")
    parser.add_argument("--num-instances", type=int, default=1)
    parser.add_argument("--baselines", default="alns,gurobi,ortools",
                        help="Comma-separated list of baselines: alns,gurobi,ortools,rl")
    parser.add_argument("--rl-model", default="",
                        help="Path to trained actor .pt file (required when 'rl' is in --baselines)")
    parser.add_argument("--arch", default="hetgat", choices=["hetgat", "simplegat"],
                        help="Model architecture of the RL checkpoint (hetgat or simplegat)")

    parser.add_argument("--n-req", type=int, default=8)
    parser.add_argument("--n-uav", type=int, default=2)
    parser.add_argument("--n-adr", type=int, default=2)
    parser.add_argument("--n-depots-uav", type=int, default=1)
    parser.add_argument("--n-depots-adr", type=int, default=1)

    parser.add_argument("--depot-sharing", action="store_true")
    parser.add_argument("--delta-minutes", type=float, default=5.0)
    parser.add_argument("--shift-minutes", type=float, default=None)

    parser.add_argument("--exact-time-limit", type=float, default=3600.0)
    parser.add_argument("--exact-mip-gap", type=float, default=0.01)
    parser.add_argument("--ortools-solver", default="SCIP")

    parser.add_argument("--alns-small-budget", type=float, default=3.0)
    parser.add_argument("--alns-large-budget", type=float, default=8.0)

    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--output-dir", default="rolling_benchmark_results")

    parser.add_argument("--demand-low",  type=float, default=1.0,
                        help="Lower bound of uniform demand distribution (kg). Default: 1.0")
    parser.add_argument("--demand-high", type=float, default=6.0,
                        help="Upper bound of uniform demand distribution (kg). Default: 6.0")
    parser.add_argument("--wind-speed-low",  type=float, default=0.0,
                        help="Min wind speed (m/s) per instance. Default: 0.0")
    parser.add_argument("--wind-speed-high", type=float, default=12.0,
                        help="Max wind speed (m/s) per instance. Default: 12.0")
    parser.add_argument("--tw-slack-mean", type=float, default=30.0,
                        help="Mean delivery time-window slack (min). Default: 30.0")
    parser.add_argument("--tw-slack-std",  type=float, default=5.0,
                        help="Std of delivery time-window slack (min). Default: 5.0")
    parser.add_argument("--tw-slack-clip-low",  type=float, default=15.0,
                        help="Min clipped delivery slack (min). Default: 15.0")
    parser.add_argument("--tw-slack-clip-high", type=float, default=60.0,
                        help="Max clipped delivery slack (min). Default: 60.0")
    parser.add_argument("--scenario-label", default="",
                        help="Label appended to output filenames and printed in the paper table header.")
    parser.add_argument("--make-plots", action="store_true", help="Generate PNG plots after benchmark export.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes for CPU baselines (alns/gurobi/ortools). "
                             "On Narval with --cpus-per-task=16, use --workers 8 with Gurobi "
                             "(leaves 2 threads per Gurobi instance). RL always runs sequentially.")

    return parser

def _print_paper_table(aggregate: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    COL = {
        'method':   18,
        'n':         6,
        'service':  14,
        'cost_req': 16,
        'tw_pu':    14,
        'tw_dl':    14,
        'uav_pct':  12,
        'time':     13,
    }

    header = (
        f"{'Method':<{COL['method']}} "
        f"{'N':>{COL['n']}} "
        f"{'Service%':>{COL['service']}} "
        f"{'Cost/req':>{COL['cost_req']}} "
        f"{'TW-PU%':>{COL['tw_pu']}} "
        f"{'TW-DL%':>{COL['tw_dl']}} "
        f"{'UAV%':>{COL['uav_pct']}} "
        f"{'Time(s)':>{COL['time']}}"
    )
    sep = '-' * len(header)

    scenario_label = getattr(args, 'scenario_label', '') or 'baseline'
    print('\n' + '=' * len(header))
    print(f'  On-demand delivery results  |  scenario={scenario_label}'
          f'  |  n_req={args.n_req}  n_uav={args.n_uav}  n_adr={args.n_adr}'
          f'  |  instances={args.num_instances}')
    print('  (values shown as mean±std across instances;  service%=delivered/revealed)')
    print('=' * len(header))
    print(header)
    print(sep)

    latex_rows = []

    for row in aggregate:
        method = str(row.get('baseline', '?'))
        n_success = int(row.get('n_success', 0))
        n_runs    = int(row.get('n_runs', 1))

        revealed  = float(row.get('total_revealed_mean') or 0)
        total_cost_mu  = float(row.get('total_cost_mean') or 0)
        total_cost_sig = float(row.get('total_cost_std')  or 0)
        cost_per_req     = total_cost_mu  / max(revealed, 1.0)
        cost_per_req_std = total_cost_sig / max(revealed, 1.0)

        service_mu  = float(row.get('delivered_rate_mean') or 0) * 100.0
        service_sig = float(row.get('delivered_rate_std')  or 0) * 100.0

        pu_ok_mu  = (1.0 - float(row.get('pickup_late_rate_mean')   or 0)) * 100.0
        pu_ok_sig =        float(row.get('pickup_late_rate_std')    or 0)  * 100.0
        dl_ok_mu  = (1.0 - float(row.get('delivery_late_rate_mean') or 0)) * 100.0
        dl_ok_sig =        float(row.get('delivery_late_rate_std')  or 0)  * 100.0

        op_total = float(row.get('operating_cost_mean') or 0)
        op_uav   = float(row.get('operating_cost_uav_mean') or 0)
        uav_pct_mu  = (op_uav / max(op_total, 1e-9)) * 100.0
        uav_pct_sig = float(row.get('operating_cost_uav_std') or 0) / max(op_total, 1e-9) * 100.0

        wall_mu  = float(row.get('wall_time_s_mean') or 0)
        wall_sig = float(row.get('wall_time_s_std')  or 0)

        def _fmt_pm(mu, sig, fmt='.1f') -> str:
            return f'{mu:{fmt}}±{sig:{fmt}}' if sig else f'{mu:{fmt}}'

        n_str = f'{n_success}/{n_runs}'
        line = (
            f'{method:<{COL["method"]}} '
            f'{n_str:>{COL["n"]}} '
            f'{_fmt_pm(service_mu, service_sig):>{COL["service"]}} '
            f'{_fmt_pm(cost_per_req, cost_per_req_std, fmt=".4f"):>{COL["cost_req"]}} '
            f'{_fmt_pm(pu_ok_mu, pu_ok_sig):>{COL["tw_pu"]}} '
            f'{_fmt_pm(dl_ok_mu, dl_ok_sig):>{COL["tw_dl"]}} '
            f'{_fmt_pm(uav_pct_mu, uav_pct_sig):>{COL["uav_pct"]}} '
            f'{_fmt_pm(wall_mu, wall_sig):>{COL["time"]}}'
        )
        print(line)

        latex_rows.append(
            f'  {method} & {n_success}/{n_runs}'
            f' & ${service_mu:.1f}\\pm{service_sig:.1f}$\\%'
            f' & ${cost_per_req:.4f}\\pm{cost_per_req_std:.4f}$'
            f' & ${pu_ok_mu:.1f}\\pm{pu_ok_sig:.1f}$\\%'
            f' & ${dl_ok_mu:.1f}\\pm{dl_ok_sig:.1f}$\\%'
            f' & ${uav_pct_mu:.1f}\\pm{uav_pct_sig:.1f}$\\%'
            f' & ${wall_mu:.1f}\\pm{wall_sig:.1f}$ \\\\'
        )

    print(sep)
    print('  NOTE: HetGAT-RL rows can be appended from evaluate.py --paper-table output.')
    print('=' * len(header))

    scenario_label = getattr(args, 'scenario_label', '') or 'baseline'
    latex_path = Path(args.output_dir) / f'paper_table_{scenario_label}.tex'
    n_config = f'$n_{{req}}={args.n_req}$, $n_{{UAV}}={args.n_uav}$, $n_{{ADR}}={args.n_adr}$'
    latex = [
        r'\begin{table}[t]',
        r'  \centering',
        r'  \caption{On-demand delivery results (' + scenario_label + r') --- ' + n_config
            + r'.  All values shown as $\mu \pm \sigma$ across instances.}',
        r'  \label{tab:results_' + scenario_label + r'}',
        r'  \begin{tabular}{lrcccccc}',
        r'    \toprule',
        r'    Method & N & Service\% & Cost/req & TW-PU\% & TW-DL\% & UAV\% & Time(s) \\',
        r'    \midrule',
    ] + latex_rows + [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]
    latex_path.write_text('\n'.join(latex), encoding='utf-8')
    print(f'  LaTeX table saved: {latex_path}')

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_benchmark(args)
    print(json.dumps(json_safe(result["aggregate"]), indent=2))
    print(f"Saved results to: {result['output_dir']}")
    if args.make_plots:
        try:
            from plot_rolling_benchmark_results import plot_all_metrics
            plot_all_metrics(Path(result["output_dir"]))
            print(f"Saved plots to: {Path(result['output_dir']) / 'plots'}")
        except Exception as exc:
            print(f"Plot generation failed: {type(exc).__name__}: {exc}")
    _print_paper_table(result["aggregate"], args)

if __name__ == "__main__":
    main()
