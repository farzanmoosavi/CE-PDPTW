"""
Logic tests for the FIFO and Offline-ALNS baselines.

These pin down the contract:
  - FIFO: serves requests in t_pickup order, picks nearest accessible
    vehicle that has capacity, no double-assignment, no leg to unrevealed.
  - Offline ALNS: runs the inner solver exactly once; subsequent calls
    return the cached plan unchanged.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from creat_vrp import create_instance
from dispatch_sim import Request, Vehicle, RollingHorizonDispatcher
from coalition import make_fleet


def _make_residual(inst, vehicles, revealed_ids, t=0.0):
    """Build a residual dict that mirrors what the dispatcher passes to solvers."""
    n_depots = int(inst['n_depots'])
    n_req = int(inst['n_req'])
    tw = inst['time_window'].squeeze(-1).numpy()

    active_requests = {}
    for rid in revealed_ids:
        active_requests[rid] = Request(
            req_id=rid,
            t_arrival=0.0,
            t_pickup=float(tw[n_depots + rid]),
            t_delivery=float(tw[n_depots + n_req + rid]),
            pickup_node=n_depots + rid,
            delivery_node=n_depots + n_req + rid,
            demand=float(inst['demand'][n_depots + rid].item()),
            status='waiting_for_pickup',
        )
    return {
        'full_instance': inst,
        'active_requests': active_requests,
        'vehicles': vehicles,
        'current_time': float(t),
    }


# ---------------------------------------------------------------------------
# FIFO tests
# ---------------------------------------------------------------------------

def test_fifo_serves_in_pickup_order():
    """Among assignable requests, FIFO must assign earliest-t_pickup first."""
    from fifo_baseline import FIFOSolver

    rng = np.random.default_rng(42)
    inst = create_instance(n_req=6, n_uav=1, n_adr=1,
                           n_depots_uav=1, n_depots_adr=1, rng=rng)
    inst['n_depots'] = 2
    inst['n_req'] = 6

    fleet = make_fleet(n_uav=1, n_adr=1, n_depots_uav=1, n_depots_adr=1)
    residual = _make_residual(inst, fleet, list(range(6)))

    solver = FIFOSolver()
    ops = solver.solve_ops(residual)

    # Collect (req_id, pickup_t) in the order FIFO appended pickups across
    # all vehicles.  For each vehicle, pickups should be in t_pickup order.
    for vid, vid_ops in ops.items():
        pickup_t_seq = []
        for rid, leg_type in vid_ops:
            if leg_type == 'pickup':
                pickup_t_seq.append(float(residual['active_requests'][rid].t_pickup))
        assert pickup_t_seq == sorted(pickup_t_seq), (
            f'vehicle {vid} pickup order {pickup_t_seq} is not sorted by t_pickup')


def test_fifo_no_double_assignment():
    """Every revealed request is assigned to at most one vehicle."""
    from fifo_baseline import FIFOSolver

    rng = np.random.default_rng(7)
    inst = create_instance(n_req=8, n_uav=2, n_adr=2,
                           n_depots_uav=1, n_depots_adr=1, rng=rng)
    inst['n_depots'] = 2
    inst['n_req'] = 8

    fleet = make_fleet(n_uav=2, n_adr=2, n_depots_uav=1, n_depots_adr=1)
    residual = _make_residual(inst, fleet, list(range(8)))

    ops = FIFOSolver().solve_ops(residual)

    seen_pickups = set()
    for vid_ops in ops.values():
        for rid, leg_type in vid_ops:
            if leg_type == 'pickup':
                assert rid not in seen_pickups, f'request {rid} assigned twice'
                seen_pickups.add(rid)


def test_fifo_respects_accessibility():
    """FIFO must never assign a request to a vehicle whose mode is blocked."""
    from fifo_baseline import FIFOSolver

    rng = np.random.default_rng(11)
    inst = create_instance(n_req=6, n_uav=2, n_adr=2,
                           n_depots_uav=1, n_depots_adr=1, rng=rng)
    inst['n_depots'] = 2
    inst['n_req'] = 6

    fleet = make_fleet(n_uav=2, n_adr=2, n_depots_uav=1, n_depots_adr=1)
    residual = _make_residual(inst, fleet, list(range(6)))

    ops = FIFOSolver().solve_ops(residual)
    n_depots = inst['n_depots']
    n_req = inst['n_req']

    x = inst['x'].numpy()
    for vid, vid_ops in ops.items():
        veh = fleet[vid]
        for rid, leg_type in vid_ops:
            req = residual['active_requests'][rid]
            for node in (req.pickup_node, req.delivery_node):
                if veh.mode == 'uav':
                    assert x[node, 8] > 0.5, (
                        f'UAV vehicle {vid} assigned to UAV-inaccessible node {node}')
                else:
                    assert x[node, 9] > 0.5, (
                        f'ADR vehicle {vid} assigned to ADR-inaccessible node {node}')


def test_fifo_capacity_gate():
    """A request whose demand alone exceeds vehicle capacity is never assigned."""
    from fifo_baseline import FIFOSolver

    rng = np.random.default_rng(3)
    inst = create_instance(n_req=5, n_uav=1, n_adr=1,
                           n_depots_uav=1, n_depots_adr=1, rng=rng)
    inst['n_depots'] = 2
    inst['n_req'] = 5

    fleet = make_fleet(n_uav=1, n_adr=1, n_depots_uav=1, n_depots_adr=1,
                       q_uav=1.0, q_adr=1.5)   # crippled fleet
    residual = _make_residual(inst, fleet, list(range(5)))

    ops = FIFOSolver().solve_ops(residual)

    for vid, vid_ops in ops.items():
        veh = fleet[vid]
        for rid, leg_type in vid_ops:
            req = residual['active_requests'][rid]
            assert float(req.demand) <= float(veh.capacity) + 1e-9, (
                f'request {rid} demand {req.demand} exceeds vehicle {vid} capacity {veh.capacity}')


def test_fifo_pickup_then_delivery_pairing():
    """Every pickup must be followed (eventually, same vehicle) by a delivery for that request."""
    from fifo_baseline import FIFOSolver

    rng = np.random.default_rng(9)
    inst = create_instance(n_req=6, n_uav=1, n_adr=1,
                           n_depots_uav=1, n_depots_adr=1, rng=rng)
    inst['n_depots'] = 2
    inst['n_req'] = 6

    fleet = make_fleet(n_uav=1, n_adr=1, n_depots_uav=1, n_depots_adr=1)
    residual = _make_residual(inst, fleet, list(range(6)))

    ops = FIFOSolver().solve_ops(residual)

    for vid, vid_ops in ops.items():
        picked = set()
        for rid, leg_type in vid_ops:
            if leg_type == 'pickup':
                picked.add(rid)
            else:   # delivery
                assert rid in picked, (
                    f'vehicle {vid} delivers req {rid} before picking it up')


# ---------------------------------------------------------------------------
# Offline ALNS tests
# ---------------------------------------------------------------------------

def test_offline_alns_runs_inner_only_once():
    """OfflineALNSSolver.solve must call inner_solver.solve exactly once."""
    from offline_alns import OfflineALNSSolver

    solver = OfflineALNSSolver(seed=1, time_budget_small_s=0.05,
                               time_budget_large_s=0.05)

    call_count = {'n': 0}
    real_solve = solver._inner.solve

    def counting_solve(residual):
        call_count['n'] += 1
        return real_solve(residual)

    solver._inner.solve = counting_solve   # type: ignore

    # Build a real ce_cpdptw_alns-style residual (different Vehicle class!)
    from ce_cpdptw_alns import (
        build_static_arrival_stream_from_instance,
        build_initial_fleet_from_instance,
        SyntheticALNSConfig,
        build_residual_instance,
        Request as CeRequest,
    )

    rng = np.random.default_rng(101)
    inst = create_instance(n_req=4, n_uav=1, n_adr=1,
                           n_depots_uav=1, n_depots_adr=1, rng=rng)
    inst['n_depots'] = 2
    inst['n_req'] = 4
    cfg = SyntheticALNSConfig(
        n_uav=1, n_adr=1, n_depots_uav=1, n_depots_adr=1,
        depot_sharing=True, delta_minutes=5.0, shift_minutes=120.0,
    )
    fleet = build_initial_fleet_from_instance(inst, cfg)
    arrivals = build_static_arrival_stream_from_instance(inst)
    requests = {
        int(a['req_id']): CeRequest(
            req_id=int(a['req_id']),
            t_arrival=float(a['t_arrival']),
            t_pickup=float(a['t_pickup']),
            t_delivery=float(a['t_delivery']),
            pickup_node=int(a['pickup_node']),
            delivery_node=int(a['delivery_node']),
            demand=float(a['demand']),
            status='waiting_for_pickup',
        )
        for a in arrivals
    }
    residual = build_residual_instance(requests=requests, vehicles=fleet,
                                       current_time=0.0, full_instance=inst)

    # Call solve() three times — only the first should hit the inner solver
    plan1 = solver.solve(residual)
    plan2 = solver.solve(residual)
    plan3 = solver.solve(residual)

    assert call_count['n'] == 1, (
        f'inner solver called {call_count["n"]} times; offline ALNS must call exactly once')

    # All three returned plans must be the same object structure
    assert set(plan1.keys()) == set(plan2.keys()) == set(plan3.keys())
    for vid in plan1:
        assert len(plan1[vid]) == len(plan2[vid]) == len(plan3[vid])


def test_offline_alns_end_to_end_runs_clean():
    """End-to-end: solve_static_instance_with_offline_alns produces an episode log."""
    from offline_alns import solve_static_instance_with_offline_alns
    from ce_cpdptw_alns import SyntheticALNSConfig

    rng = np.random.default_rng(55)
    inst = create_instance(n_req=5, n_uav=2, n_adr=2,
                           n_depots_uav=1, n_depots_adr=1, rng=rng)
    inst['n_depots'] = 2
    inst['n_req'] = 5
    cfg = SyntheticALNSConfig(
        n_uav=2, n_adr=2, n_depots_uav=1, n_depots_adr=1,
        depot_sharing=True, delta_minutes=5.0, shift_minutes=120.0,
    )
    log = solve_static_instance_with_offline_alns(
        inst, cfg, seed=42,
        time_budget_small_s=0.2, time_budget_large_s=0.2,
    )
    assert isinstance(log, list) and len(log) > 0
    # Final log entry should include a summary block when the shift ends
    last = log[-1]
    assert 'summary' in last or 't' in last   # at minimum, a time entry


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
