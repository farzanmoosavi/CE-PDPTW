from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional

def _episode_summary(log: List[Dict]) -> Dict[str, Any]:
    return next((e for e in log if e.get('summary')), {})

def _extract_request_history(episode_logs: List[List[Dict]]) -> List[Dict]:
    history: List[Dict] = []
    for log in episode_logs:
        history.extend(_episode_summary(log).get('request_history', []))
    return history

def _pickup_target(req: Dict[str, Any]) -> Optional[float]:
    return req.get('t_pickup_target', req.get('t_p'))

def _delivery_target(req: Dict[str, Any]) -> Optional[float]:
    return req.get('t_delivery_target', req.get('t_d'))

def compute_feasibility_rate(episode_logs: List[List[Dict]]) -> float:
    total = 0
    feasible = 0
    for req in _extract_request_history(episode_logs):
        total += 1
        t_pick = _pickup_target(req)
        t_del = _delivery_target(req)
        a_pick = req.get('T_pickup_actual')
        a_del = req.get('T_delivery_actual')
        if (
            a_pick is not None and
            a_del is not None and
            t_pick is not None and
            t_del is not None and
            a_pick <= t_pick and
            a_del <= t_del
        ):
            feasible += 1
    return feasible / max(total, 1)

def compute_violation_magnitude(requests_history: List[Dict]) -> Dict[str, float]:
    late_pickups = []
    late_deliveries = []

    for req in requests_history:
        t_pick = _pickup_target(req)
        t_del = _delivery_target(req)
        a_pick = req.get('T_pickup_actual')
        a_del = req.get('T_delivery_actual')

        if a_pick is not None and t_pick is not None and a_pick > t_pick:
            late_pickups.append(a_pick - t_pick)
        if a_del is not None and t_del is not None and a_del > t_del:
            late_deliveries.append(a_del - t_del)

    return {
        'mean_late_pickup_min': float(np.mean(late_pickups)) if late_pickups else 0.0,
        'mean_late_delivery_min': float(np.mean(late_deliveries)) if late_deliveries else 0.0,
        'pct_late_pickup': len(late_pickups) / max(len(requests_history), 1),
        'pct_late_delivery': len(late_deliveries) / max(len(requests_history), 1),
    }

def compute_battery_violations(
    vehicle_histories: Optional[List[Dict]] = None,
    episode_logs: Optional[List[List[Dict]]] = None,
    min_threshold_frac: float = 0.10,
) -> float:
    if vehicle_histories:
        total, violations = 0, 0
        for vh in vehicle_histories:
            if vh.get('route_end'):
                total += 1
                if vh['battery_end'] < vh['battery_init'] * min_threshold_frac:
                    violations += 1
        return violations / max(total, 1)

    if episode_logs:
        summaries = [_episode_summary(log) for log in episode_logs]
        hits = sum(1 for s in summaries if float(s.get('battery_penalty_cost', 0.0)) > 0.0)
        return hits / max(len(summaries), 1)

    return 0.0

def compute_runtime_stats(episode_logs: List[List[Dict]]) -> Dict[str, float]:
    per_decision = []
    per_shift = []

    for log in episode_logs:
        shift_t = 0.0
        for entry in log:
            if entry.get('summary'):
                continue
            st = float(entry.get('solver_time_s', 0.0))
            per_decision.append(st)
            shift_t += st
        per_shift.append(shift_t)

    pd_arr = np.array(per_decision) if per_decision else np.array([0.0])
    ps_arr = np.array(per_shift) if per_shift else np.array([0.0])
    return {
        'per_decision_median_s': float(np.median(pd_arr)),
        'per_decision_p95_s': float(np.percentile(pd_arr, 95)),
        'per_shift_mean_s': float(ps_arr.mean()),
        'per_shift_std_s': float(ps_arr.std()),
    }

def amortized_inference_time(
    training_time_s: float,
    n_eval_shifts: int,
    mean_requests_per_shift: float,
) -> float:
    total_requests = n_eval_shifts * mean_requests_per_shift
    if total_requests <= 0:
        return float('inf')
    return training_time_s / total_requests

def make_report_row(
    method_name: str,
    delta_min: float,
    n_req: int,
    episode_logs: List[List[Dict]],
    requests_history: Optional[List[Dict]] = None,
    vehicle_histories: Optional[List[Dict]] = None,
    training_time_s: float = 0.0,
) -> Dict[str, Any]:
    feas = compute_feasibility_rate(episode_logs)
    rt = compute_runtime_stats(episode_logs)
    n_eps = len(episode_logs)
    mean_requests = np.mean([
        _episode_summary(log).get('total_revealed', 0)
        for log in episode_logs
    ]) if episode_logs else 1.0

    row = {
        'method': method_name,
        'delta_min': delta_min,
        'n_req': n_req,
        'feasibility_rate': feas,
        'per_decision_median_s': rt['per_decision_median_s'],
        'per_decision_p95_s': rt['per_decision_p95_s'],
        'per_shift_mean_s': rt['per_shift_mean_s'],
        'training_time_s': training_time_s,
        'amortized_inference_s': amortized_inference_time(
            training_time_s, n_eps, mean_requests
        ),
    }

    history = requests_history if requests_history is not None else _extract_request_history(episode_logs)
    if history:
        row.update(compute_violation_magnitude(history))

    if vehicle_histories:
        row['battery_violation_rate'] = compute_battery_violations(vehicle_histories=vehicle_histories)
    else:
        row['battery_violation_rate'] = compute_battery_violations(episode_logs=episode_logs)

    return row

def print_report(rows: List[Dict]):
    keys = [
        'method', 'delta_min', 'n_req', 'feasibility_rate',
        'pct_late_pickup', 'pct_late_delivery',
        'battery_violation_rate',
        'per_decision_median_s', 'per_shift_mean_s',
        'training_time_s', 'amortized_inference_s',
    ]
    header = '  '.join(f'{k:>24}' for k in keys)
    print(header)
    print('-' * len(header))
    for row in rows:
        line = '  '.join(f'{str(row.get(k, ""))[:24]:>24}' for k in keys)
        print(line)
