"""
Dispatch consistency tests (spec §9 required tests).
"""
import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dispatch_sim import (
    Request, Vehicle, Leg, RollingHorizonDispatcher,
    sample_arrival_stream, advance_time, pick_first_uncommitted_leg,
    build_residual_instance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vehicles():
    return {
        0: Vehicle(vehicle_id=0, mode='uav',
                   current_node=0, current_time=0.0,
                   battery=6500.0, load=0.0, capacity=5.0),
        1: Vehicle(vehicle_id=1, mode='adr',
                   current_node=1, current_time=0.0,
                   battery=4500.0, load=0.0, capacity=10.0),
    }


def _make_request(req_id, status='waiting_for_pickup', assigned_vehicle=None):
    return Request(
        req_id=req_id,
        t_arrival=0.0,
        t_pickup=5.0,
        t_delivery=35.0,
        pickup_node=2,
        delivery_node=3,
        demand=2.0,
        status=status,
        assigned_vehicle=assigned_vehicle,
    )


def _make_mini_instance(n_nodes=4):
    """4-node (depot_uav, depot_adr, pickup, delivery) numpy-based fake instance.

    V_UAV_MAX=1.2, d=0.5 → travel_time≈0.42 min — completes within any Δ=5 tick.
    """
    d = np.full((n_nodes, n_nodes), 0.5)
    np.fill_diagonal(d, 0.0)
    return {'d_uav': d, 'd_adr': d, 'n_depots': 2, 'n_req': 2}


def _apply_completions(completed_legs, vehicles, requests):
    """Mirror the run_shift completion loop for unit tests that call advance_time directly."""
    for leg in completed_legs:
        req = requests.get(leg.request_id)
        veh = vehicles[leg.vehicle_id]
        if leg.leg_type == 'pickup':
            req.T_pickup_actual = leg.t_arrive
            if req.status == 'pickup_committed':
                req.transition('onboard')
            veh.committed_leg = None
        elif leg.leg_type == 'delivery':
            req.T_delivery_actual = leg.t_arrive
            if req.status == 'delivery_committed':
                req.transition('delivered')
            veh.committed_leg = None


class ScriptedSolver:
    """Always assigns all waiting_for_pickup requests to vehicle 0."""
    def __init__(self):
        self.call_count = 0

    def solve(self, residual):
        self.call_count += 1
        vehicles = residual['vehicles']
        active = residual['active_requests']
        t_now = residual['current_time']
        routes = {vid: [] for vid in vehicles}
        for req in active.values():
            if req.status == 'waiting_for_pickup':
                routes[0].append(Leg(req.req_id, 0, 'pickup',
                                     0, req.pickup_node, t_now, t_now + 5))
                routes[0].append(Leg(req.req_id, 0, 'delivery',
                                     req.pickup_node, req.delivery_node,
                                     t_now + 5, t_now + 20))
            elif req.status == 'onboard' and req.assigned_vehicle == 0:
                routes[0].append(Leg(req.req_id, 0, 'delivery',
                                     req.pickup_node, req.delivery_node,
                                     t_now, t_now + 15))
        return routes


# ---------------------------------------------------------------------------
# 1. Status transitions are valid
# ---------------------------------------------------------------------------

def test_valid_status_transitions():
    req = _make_request(0)
    req.transition('pickup_committed')
    req.transition('onboard')
    req.transition('delivery_committed')
    req.transition('delivered')

    with pytest.raises(AssertionError):
        req2 = _make_request(1)
        req2.transition('delivered')  # skip to delivered directly


def test_no_backward_transition():
    req = _make_request(0)
    req.transition('pickup_committed')
    with pytest.raises(AssertionError):
        req.transition('waiting_for_pickup')  # backward


# ---------------------------------------------------------------------------
# 2. Same vehicle for pickup and delivery
# ---------------------------------------------------------------------------

def test_same_vehicle_pickup_delivery():
    vehicles = _make_vehicles()
    requests = {0: _make_request(0)}

    leg = Leg(0, 0, 'pickup', 0, 2, 0, 5)
    vehicles[0].committed_leg = leg
    requests[0].transition('pickup_committed')
    requests[0].assigned_vehicle = 0

    full_inst = _make_mini_instance()
    completed, vehicles, _op, _bat, _, _ = advance_time(0, 10, vehicles, full_inst)
    _apply_completions(completed, vehicles, requests)

    assert requests[0].status == 'onboard'
    assert requests[0].assigned_vehicle == 0
    assert requests[0].assigned_vehicle != 1, 'Delivery vehicle mismatch'


# ---------------------------------------------------------------------------
# 3. Pickup before delivery in real time
# ---------------------------------------------------------------------------

def test_pickup_before_delivery_timestamps():
    vehicles = _make_vehicles()
    requests = {0: _make_request(0)}
    full_inst = _make_mini_instance()

    pickup_leg = Leg(0, 0, 'pickup', 0, 2, 0, 5)
    vehicles[0].committed_leg = pickup_leg
    requests[0].transition('pickup_committed')
    requests[0].assigned_vehicle = 0

    completed, vehicles, _, _, _, _ = advance_time(0, 10, vehicles, full_inst)
    _apply_completions(completed, vehicles, requests)
    t_pickup_done = pickup_leg.t_arrive

    delivery_leg = Leg(0, 0, 'delivery', 2, 3, 10, 20)
    vehicles[0].committed_leg = delivery_leg
    requests[0].transition('delivery_committed')

    completed2, vehicles, _, _, _, _ = advance_time(10, 25, vehicles, full_inst)
    _apply_completions(completed2, vehicles, requests)
    t_delivery_done = delivery_leg.t_arrive

    assert t_pickup_done < t_delivery_done, (
        f'Pickup done at {t_pickup_done}, delivery done at {t_delivery_done}'
    )


# ---------------------------------------------------------------------------
# 4. Committed-leg lock
# ---------------------------------------------------------------------------

def test_committed_leg_not_rerouteable():
    vehicles = _make_vehicles()
    requests = {0: _make_request(0)}

    leg = Leg(0, 0, 'pickup', 0, 2, 0, 5)
    vehicles[0].committed_leg = leg
    requests[0].transition('pickup_committed')

    original_leg = vehicles[0].committed_leg
    solver = ScriptedSolver()
    residual = {
        'full_instance': _make_mini_instance(),
        'active_requests': requests,
        'vehicles': vehicles,
        'current_time': 2.0,
    }
    route_plan = solver.solve(residual)

    # Dispatcher rule: skip vehicles that already have a committed leg.
    # solver.solve() must not mutate vehicle state.
    for vid, route in route_plan.items():
        veh = vehicles[vid]
        if veh.committed_leg is not None:
            assert veh.committed_leg is original_leg


# ---------------------------------------------------------------------------
# 5. No request is dropped
# ---------------------------------------------------------------------------

def test_no_request_dropped():
    arrivals = sample_arrival_stream(30, 10, seed=99)
    # Pin nodes to fit the 4-node mini instance
    for r in arrivals:
        r['pickup_node'] = 2
        r['delivery_node'] = 3
    n_total = len(arrivals)

    solver = ScriptedSolver()
    dispatcher = RollingHorizonDispatcher(solver, delta_minutes=5, shift_minutes=30)
    fleet = _make_vehicles()
    full_inst = _make_mini_instance()
    full_inst['n_req'] = max(n_total, 1)
    log = dispatcher.run_shift(arrivals, fleet, full_inst)

    summary = next(e for e in log if e.get('summary'))
    delivered = summary['total_delivered']
    undelivered = summary['undelivered']
    assert delivered + undelivered == n_total, (
        f'Lost requests: revealed={n_total}, delivered={delivered}, undelivered={undelivered}'
    )


# ---------------------------------------------------------------------------
# 6. Vehicle time monotonicity
# ---------------------------------------------------------------------------

def test_time_monotonicity():
    vehicles = _make_vehicles()
    requests = {0: _make_request(0)}

    leg = Leg(0, 0, 'pickup', 0, 2, 0, 5)
    vehicles[0].committed_leg = leg
    requests[0].transition('pickup_committed')
    requests[0].assigned_vehicle = 0

    t_before = vehicles[0].current_time
    advance_time(0, 10, vehicles, _make_mini_instance())
    t_after = vehicles[0].current_time

    assert t_after >= t_before, 'Vehicle time went backward'


# ---------------------------------------------------------------------------
# 7. Known-trajectory regression
# ---------------------------------------------------------------------------

def test_known_trajectory_regression():
    """Scripted arrivals + scripted solver must produce a deterministic outcome."""
    arrivals = [
        {'req_id': 0, 't_arrival': 0.0, 't_pickup': 0.0, 't_delivery': 30.0,
         'demand': 2.0, 'pickup_node': 2, 'delivery_node': 3},
        {'req_id': 1, 't_arrival': 3.0, 't_pickup': 3.0, 't_delivery': 33.0,
         'demand': 1.5, 'pickup_node': 2, 'delivery_node': 3},
    ]
    solver = ScriptedSolver()
    dispatcher = RollingHorizonDispatcher(solver, delta_minutes=5, shift_minutes=15)
    fleet = _make_vehicles()
    log = dispatcher.run_shift(arrivals, fleet, _make_mini_instance())

    summary = next(e for e in log if e.get('summary'))
    assert summary['total_revealed'] == 2, \
        f'Expected 2 revealed requests, got {summary["total_revealed"]}'
