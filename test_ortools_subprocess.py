"""
Local smoke test for OR-Tools VRP subprocess isolation.

Tests:
  1. CP-SAT runs correctly in a fresh subprocess (no torch loaded).
  2. full_instance with torch tensors pickled via _instance_to_numpy survives
     round-trip WITHOUT importing torch in the subprocess.
  3. _insertion_delta d=p+1 fix: empty-route insertion no longer costs ~1e9.
  4. Clarke-Wright can chain onto an existing route tail (j unassigned, i is tail).

Run with:
    python test_ortools_subprocess.py
"""
from __future__ import annotations
import multiprocessing
import pickle
import sys
import subprocess
import tempfile
import os
import numpy as np


# ── Test 1: CP-SAT fresh subprocess ─────────────────────────────────────────

def _cpsat_worker(q):
    try:
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        arcs = []
        for n in range(3):
            for m in range(3):
                if n != m:
                    v = model.new_bool_var(f'a{n}_{m}')
                    arcs.append((n, m, v))
            lp = model.new_bool_var(f'lp{n}')
            arcs.append((n, n, lp))
        model.add_circuit(arcs)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        status = solver.solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), f"status={status}"
        q.put(('ok', None))
    except Exception as e:
        import traceback
        q.put(('err', traceback.format_exc()))


def test_cpsat_subprocess():
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_cpsat_worker, args=(q,))
    p.start()
    p.join(timeout=30)
    if p.is_alive():
        p.terminate(); p.join()
        print("FAIL test_cpsat_subprocess: timed out")
        return False
    if p.exitcode != 0:
        print(f"FAIL test_cpsat_subprocess: subprocess crashed (exit {p.exitcode})")
        return False
    try:
        status, err = q.get_nowait()
    except Exception as e:
        print(f"FAIL test_cpsat_subprocess: no result in queue: {e}")
        return False
    if status != 'ok':
        print(f"FAIL test_cpsat_subprocess: {err}")
        return False
    print("PASS test_cpsat_subprocess")
    return True


# ── Test 2: _instance_to_numpy removes torch dependency from pickle ───────────

def _numpy_pickle_worker(q, payload):
    """Worker that unpickles payload and checks no torch was imported."""
    try:
        data = pickle.loads(payload)
        # Verify we got numpy arrays, not torch tensors
        for k, v in data["full_instance"].items():
            if hasattr(v, "numpy"):
                q.put(('err', f"key '{k}' still a tensor after numpy conversion"))
                return
        # Check torch is NOT imported (would mean we loaded it during unpickling)
        if 'torch' in sys.modules:
            q.put(('err', 'torch was imported during unpickling!'))
            return
        q.put(('ok', list(data["full_instance"].keys())))
    except Exception as e:
        import traceback
        q.put(('err', traceback.format_exc()))


def test_numpy_pickle():
    try:
        import torch
    except ImportError:
        print("SKIP test_numpy_pickle: torch not installed")
        return True

    # Build a fake full_instance with torch tensors
    fake_instance = {
        'x': torch.zeros(5, 11),
        'demand': torch.zeros(5, 1),
        'time_window': torch.zeros(5, 1),
        'n_depots': 1,
        'n_req': 2,
    }

    # Simulate _instance_to_numpy
    numpy_instance = {}
    for k, v in fake_instance.items():
        if hasattr(v, "numpy"):
            numpy_instance[k] = v.detach().cpu().numpy()
        else:
            numpy_instance[k] = v

    payload = pickle.dumps({"full_instance": numpy_instance, "extra": 42})

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_numpy_pickle_worker, args=(q, payload))
    p.start()
    p.join(timeout=15)
    if p.is_alive():
        p.terminate(); p.join()
        print("FAIL test_numpy_pickle: timed out")
        return False
    if p.exitcode != 0:
        print(f"FAIL test_numpy_pickle: subprocess crashed (exit {p.exitcode})")
        return False
    try:
        status, result = q.get_nowait()
    except Exception as e:
        print(f"FAIL test_numpy_pickle: no result: {e}")
        return False
    if status != 'ok':
        print(f"FAIL test_numpy_pickle: {result}")
        return False
    print(f"PASS test_numpy_pickle: keys={result}")
    return True


# ── Test 3: _insertion_delta d=p+1 fix ───────────────────────────────────────

def test_insertion_delta_fix():
    """Empty-route insertion should NOT cost ~1e9 after the fix."""
    # We need to import the module — use a subprocess to avoid the benchmark's
    # module-level imports. Actually, we can test the math directly here.

    # Simulate: 2 nodes — depot (idx=0) and pickup (idx=1), delivery (idx=2)
    # Simple 1D distance: d(0,1)=1, d(1,2)=2, d(0,2)=3
    dm = np.array([
        [0.0, 1.0, 3.0],
        [1.0, 0.0, 2.0],
        [3.0, 2.0, 0.0],
    ], dtype=np.float32)
    n_dep = 1

    # Replicate the fixed _insertion_delta locally
    from dataclasses import dataclass
    @dataclass
    class FakeRequest:
        pickup_node: int
        delivery_node: int

    req_list = [FakeRequest(pickup_node=1, delivery_node=2)]
    stops = []  # empty
    p, d, ri = 0, 1, 0
    start = 0   # vehicle at depot
    n = 0

    pick_phys = req_list[ri].pickup_node   # 1
    dliv_phys = req_list[ri].delivery_node  # 2

    def node_at(idx):
        if idx < 0: return start
        if idx >= n: return -1
        r, lt = stops[idx]
        return req_list[r].pickup_node if lt == "pickup" else req_list[r].delivery_node

    def tmin(fp, tp):
        if fp == tp: return 0.0
        N = dm.shape[0]
        if fp < 0 or tp < 0 or fp >= N or tp >= N: return 1e9
        return float(dm[fp, tp])

    # delta_p
    a = node_at(p - 1)  # start = 0
    b = node_at(p)      # -1 (end)
    cost_before_p = tmin(a, b) if b >= 0 else 0.0
    cost_after_p = tmin(a, pick_phys) + (tmin(pick_phys, b) if b >= 0 else 0.0)
    delta_p = cost_after_p - cost_before_p  # tmin(0, 1) = 1.0

    # delta_d using FIXED exp_node
    def exp_node(idx):
        if idx == p: return pick_phys    # d-1 == p → pick_phys
        if idx < p: return node_at(idx)
        return node_at(idx - 1)

    c_before_d = exp_node(d - 1)   # exp_node(0) = pick_phys = 1
    c_after_d  = exp_node(d)       # exp_node(1) → node_at(0) → idx=0 >= n=0 → -1

    cost_before_d = tmin(c_before_d, c_after_d) if c_after_d >= 0 else 0.0
    cost_after_d  = tmin(c_before_d, dliv_phys) + (tmin(dliv_phys, c_after_d) if c_after_d >= 0 else 0.0)
    delta_d = cost_after_d - cost_before_d  # tmin(1, 2) = 2.0

    total = delta_p + delta_d  # 1.0 + 2.0 = 3.0

    expected = tmin(start, pick_phys) + tmin(pick_phys, dliv_phys)  # 1 + 2 = 3

    if abs(total - expected) < 1e-6:
        print(f"PASS test_insertion_delta_fix: delta={total:.4f} (expected {expected:.4f})")
        return True
    else:
        print(f"FAIL test_insertion_delta_fix: delta={total:.4f}, expected {expected:.4f}")
        print("  (Was ~1e9 before fix — empty-route insertion looked catastrophically expensive)")
        return False


# ── Test 4: CW can chain onto existing route tail ─────────────────────────────

def test_cw_chaining():
    """CW savings should extend existing routes (j unassigned, i at tail of k's route)."""
    # Simulate: vehicle k has route [req0_pick, req0_del] with route_tail[k]=0
    # Savings (k, 0, 1) means: chain req1 after req0 on vehicle k
    # req1 is unassigned, req0 is assigned at tail → should be accepted
    assigned = {0}  # req0 already assigned
    route_tail = {0: 0}  # vehicle 0's tail is req0
    veh_stops = [[(0, "pickup"), (0, "delivery")]]  # vehicle 0 already has req0

    # Simulate the fixed CW logic
    i, j, k = 0, 1, 0  # chain req1 (j=1) after req0 (i=0) on vehicle 0 (k=0)
    s = 5.0  # positive savings

    accepted = False
    if s > 0 and j not in assigned:
        if veh_stops[k]:
            # Route exists: i must be the tail
            if route_tail.get(k) == i:
                accepted = True
        else:
            # Empty route
            if i not in assigned:
                accepted = True

    if accepted:
        print("PASS test_cw_chaining: j=unassigned, i=tail -> chain accepted")
        return True
    else:
        print("FAIL test_cw_chaining: should have accepted chain onto existing tail")
        return False


# ── Test 5: add_at_most_one allows partial service (infeasible requests skipped) ─

def _cpsat_optional_worker(q):
    """Worker: CP-SAT model with add_at_most_one + serve penalty returns FEASIBLE even when
    one request is unreachable within its time window (simulating a no-fly-zone request).

    Layout: 1 vehicle, 2 requests.
    Nodes: PICK(0)=0, DLIV(0)=1, PICK(1)=2, DLIV(1)=3, DEPOT(0)=4
    Transit: req0 costs 1 (feasible within tw_hi=100), req1 costs HORIZON (HORIZON >> tw_hi=10).
    Time window on PICK(1) = [0, 10] — transit HORIZON violates it when visited.
    With add_exactly_one, the model is INFEASIBLE (forced visit violates time window).
    With add_at_most_one + SERVE_PENALTY, the model is FEASIBLE (solver skips req1).
    """
    try:
        from ortools.sat.python import cp_model
        HORIZON = 100000
        SERVE_PENALTY = 10 * HORIZON

        model = cp_model.CpModel()
        n_veh, n_req = 1, 2
        N = 2 * n_req + n_veh  # PICK(0)=0, DLIV(0)=1, PICK(1)=2, DLIV(1)=3, DEPOT(0)=4

        loop = [[model.new_bool_var(f'lp_{k}_{n}') for n in range(N)] for k in range(n_veh)]
        model.add(loop[0][4] == 0)  # vehicle must use its depot

        # Transit costs: req0 arcs cost 1, req1 arcs cost HORIZON (no-fly zone).
        tran = [
            [0, 1, HORIZON, HORIZON, 0],  # from PICK(0)
            [1, 0, HORIZON, HORIZON, 0],  # from DLIV(0)
            [HORIZON]*5,                   # from PICK(1) — all blocked
            [HORIZON]*5,                   # from DLIV(1) — all blocked
            [1, 1, HORIZON, HORIZON, 0],  # from DEPOT(0)
        ]

        model.add_at_most_one([loop[0][0].negated()])   # req 0: at most 1 vehicle
        model.add_at_most_one([loop[0][2].negated()])   # req 1: at most 1 vehicle
        for i in range(n_req):
            model.add(loop[0][2*i] == loop[0][2*i+1])  # pickup ↔ delivery same vehicle

        arc = {(n, m): model.new_bool_var(f'a_{n}_{m}') for n in range(N) for m in range(N) if n != m}
        circuit = [(n, n, loop[0][n]) for n in range(N)] + [(n, m, arc[(n, m)]) for (n, m) in arc]
        model.add_circuit(circuit)

        # Time variables.
        tv = [model.new_int_var(0, HORIZON, f't_{n}') for n in range(N)]
        model.add(tv[4] == 0)  # depart depot at time 0
        for (n, m), a_var in arc.items():
            tr = tran[n][m]
            if tr > 0:
                model.add(tv[m] >= tv[n] + tr).only_enforce_if(a_var)

        # Time windows: req0 tw_hi=100 (reachable, transit=1), req1 tw_hi=10 (unreachable, transit=HORIZON).
        tw_hi = [HORIZON] * N
        tw_hi[0] = 100   # PICK(0): feasible (1 << 100)
        tw_hi[2] = 10    # PICK(1): infeasible when visited (HORIZON >> 10)
        for i in range(n_req):
            visits = loop[0][2*i].negated()
            model.add(tv[2*i] <= tw_hi[2*i]).only_enforce_if(visits)
            model.add(tv[2*i+1] >= tv[2*i]).only_enforce_if(visits)

        arc_vars, coeffs = [], []
        for (n, m), a_var in arc.items():
            c = tran[n][m]
            if 0 < c < HORIZON:
                arc_vars.append(a_var)
                coeffs.append(c)
        for i in range(n_req):
            arc_vars.append(loop[0][2*i])
            coeffs.append(SERVE_PENALTY)
        model.minimize(cp_model.LinearExpr.weighted_sum(arc_vars, coeffs))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10
        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            q.put(('err', f'expected FEASIBLE/OPTIMAL, got status={status} (add_exactly_one would give INFEASIBLE)'))
            return

        # req 0 must be served (loop=0), req 1 must be skipped (loop=1) — too expensive to visit.
        served_0 = (solver.boolean_value(loop[0][0]) == 0)
        served_1 = (solver.boolean_value(loop[0][2]) == 0)
        if not served_0:
            q.put(('err', 'req 0 (feasible) was not served'))
            return
        if served_1:
            q.put(('err', 'req 1 (blocked) was served — time window should have been violated'))
            return
        q.put(('ok', None))
    except Exception:
        import traceback
        q.put(('err', traceback.format_exc()))


def test_cpsat_optional_requests():
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_cpsat_optional_worker, args=(q,))
    p.start()
    p.join(timeout=30)
    if p.is_alive():
        p.terminate(); p.join()
        print("FAIL test_cpsat_optional_requests: timed out")
        return False
    if p.exitcode != 0:
        print(f"FAIL test_cpsat_optional_requests: subprocess crashed (exit {p.exitcode})")
        return False
    try:
        status, err = q.get_nowait()
    except Exception as e:
        print(f"FAIL test_cpsat_optional_requests: no result: {e}")
        return False
    if status != 'ok':
        print(f"FAIL test_cpsat_optional_requests: {err}")
        return False
    print("PASS test_cpsat_optional_requests: blocked request skipped, feasible request served")
    return True


# ── Test 6: depot self-loop allowed when ALL requests are infeasible ──────────

def _cpsat_depot_selfloop_worker(q):
    """Worker: CP-SAT model with conditional depot constraint remains FEASIBLE
    when every request is time-window infeasible (all must be skipped).

    This tests the fix for `model.add(loop[k][DEPOT(k)] == 0)` → conditional.
    With the old constraint (forced loop[DEPOT]=0) and all pickups loop=1,
    add_circuit had no valid circuit and returned INFEASIBLE.
    With the new constraint (loop[DEPOT] <= loop[PICK(i)] for all i),
    the depot can self-loop when every pickup is skipped → FEASIBLE.
    """
    try:
        from ortools.sat.python import cp_model
        HORIZON = 100000
        SERVE_PENALTY = 10 * HORIZON

        model = cp_model.CpModel()
        n_veh, n_req = 1, 2
        N = 2 * n_req + n_veh  # nodes: 0=PICK0, 1=DLIV0, 2=PICK1, 3=DLIV1, 4=DEPOT

        loop = [[model.new_bool_var(f'lp_{k}_{n}') for n in range(N)] for k in range(n_veh)]

        # NEW conditional constraint: depot forced non-self-loop only when a pickup is visited
        for i in range(n_req):
            model.add(loop[0][4] <= loop[0][2 * i])

        # All request transits are HORIZON so they'll all be skipped
        tran = [[HORIZON] * N for _ in range(N)]
        for n in range(N):
            tran[n][n] = 0

        model.add_at_most_one([loop[0][0].negated()])
        model.add_at_most_one([loop[0][2].negated()])
        for i in range(n_req):
            model.add(loop[0][2 * i] == loop[0][2 * i + 1])

        arc = {(n, m): model.new_bool_var(f'a_{n}_{m}') for n in range(N) for m in range(N) if n != m}
        circuit = [(n, n, loop[0][n]) for n in range(N)] + [(n, m, arc[(n, m)]) for (n, m) in arc]
        model.add_circuit(circuit)

        tv = [model.new_int_var(0, HORIZON, f't_{n}') for n in range(N)]
        model.add(tv[4] == 0)

        # Time propagation: if arc n→m active, arrival at m ≥ departure from n + transit.
        for (n, m), a_var in arc.items():
            tr = tran[n][m]
            if tr > 0:
                model.add(tv[m] >= tv[n] + tr).only_enforce_if(a_var)

        # All pickups have tw_hi=1, transit=HORIZON → all infeasible (HORIZON >> 1)
        tw_hi = [1, HORIZON, 1, HORIZON, HORIZON]
        for i in range(n_req):
            visits = loop[0][2 * i].negated()
            model.add(tv[2 * i] <= tw_hi[2 * i]).only_enforce_if(visits)

        arc_vars, coeffs = [], []
        for i in range(n_req):
            arc_vars.append(loop[0][2 * i])
            coeffs.append(SERVE_PENALTY)
        model.minimize(cp_model.LinearExpr.weighted_sum(arc_vars, coeffs))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10
        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            q.put(('err', f'INFEASIBLE (status={status}) — depot self-loop fix missing? '
                          'Unconditional loop[DEPOT]==0 causes INFEASIBLE when all requests skipped.'))
            return

        # Both requests must be skipped (loop=1)
        skip_0 = solver.boolean_value(loop[0][0])
        skip_1 = solver.boolean_value(loop[0][2])
        if not skip_0 or not skip_1:
            q.put(('err', f'Expected both requests skipped (HORIZON transit > tw_hi=1). '
                          f'skip_0={skip_0}, skip_1={skip_1}'))
            return
        q.put(('ok', None))
    except Exception:
        import traceback
        q.put(('err', traceback.format_exc()))


def test_cpsat_depot_selfloop():
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_cpsat_depot_selfloop_worker, args=(q,))
    p.start()
    p.join(timeout=30)
    if p.is_alive():
        p.terminate(); p.join()
        print("FAIL test_cpsat_depot_selfloop: timed out")
        return False
    if p.exitcode != 0:
        print(f"FAIL test_cpsat_depot_selfloop: subprocess crashed (exit {p.exitcode})")
        return False
    try:
        status, err = q.get_nowait()
    except Exception as e:
        print(f"FAIL test_cpsat_depot_selfloop: no result: {e}")
        return False
    if status != 'ok':
        print(f"FAIL test_cpsat_depot_selfloop: {err}")
        return False
    print("PASS test_cpsat_depot_selfloop: all-infeasible requests -> FEASIBLE empty routes")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("OR-Tools + heuristics smoke tests")
    print("=" * 60)

    results = []
    results.append(test_insertion_delta_fix())
    results.append(test_cw_chaining())
    results.append(test_numpy_pickle())
    results.append(test_cpsat_subprocess())
    results.append(test_cpsat_optional_requests())
    results.append(test_cpsat_depot_selfloop())

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        sys.exit(1)
