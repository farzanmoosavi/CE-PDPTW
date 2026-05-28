from __future__ import annotations
from typing import Dict, List, Any

from ce_cpdptw_alns import ALNSBaseline, Leg

class OfflineALNSSolver:

    def __init__(
        self,
        *,
        seed: int = 42,
        time_budget_small_s: float = 3.0,
        time_budget_large_s: float = 8.0,
    ):
        self._inner = ALNSBaseline(
            seed=seed,
            time_budget_small_s=time_budget_small_s,
            time_budget_large_s=time_budget_large_s,
        )
        self._cached_plan: Dict[int, List[Leg]] | None = None

    def solve(self, residual: dict) -> Dict[int, List[Leg]]:
        if self._cached_plan is None:
            self._cached_plan = self._inner.solve(residual)
        plan = dict(self._cached_plan)
        for vid in residual['vehicles'].keys():
            plan.setdefault(vid, [])
        return plan

def solve_static_instance_with_offline_alns(
    full_instance: Dict[str, Any],
    config,
    *,
    seed: int = 42,
    time_budget_small_s: float = 3.0,
    time_budget_large_s: float = 8.0,
):
    from ce_cpdptw_alns import (
        build_dynamic_arrival_stream_from_instance,
        build_initial_fleet_from_instance,
        infer_shift_minutes,
        RollingHorizonDispatcher,
    )

    solver = OfflineALNSSolver(
        seed=seed,
        time_budget_small_s=time_budget_small_s,
        time_budget_large_s=time_budget_large_s,
    )
    arrival_stream = build_dynamic_arrival_stream_from_instance(full_instance)
    fleet = build_initial_fleet_from_instance(full_instance, config)

    sm = config.shift_minutes
    if sm is None:
        sm = infer_shift_minutes(full_instance)

    dispatcher = RollingHorizonDispatcher(
        solver=solver,
        delta_minutes=config.delta_minutes,
        shift_minutes=sm,
        alpha_p_ratio=config.alpha_p_ratio,
    )
    return dispatcher.run_shift(
        arrival_stream=arrival_stream,
        fleet_init=fleet,
        full_instance=full_instance,
    )
