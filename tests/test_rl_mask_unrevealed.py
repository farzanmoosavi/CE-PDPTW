"""
Verify _mask_unrevealed_nodes closes the information leak between the RL
solver and the rolling-horizon dispatcher.

The contract is: at each Δ-step the RL solver must produce a route plan that
depends ONLY on (a) depots and (b) currently-revealed customer nodes — exactly
the information ALNS/Gurobi/OR-Tools receive via build_residual_instance().

The strongest test is *invariance*: take a real instance, perturb the
unrevealed nodes' coordinates / demand / time windows / edge features
arbitrarily, run both versions through the masking + encoder, and confirm the
encoder produces identical outputs for the revealed subset.
"""
import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from creat_vrp import create_instance
from VRP_Actor import Model
from dispatch_sim import Request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_active_requests(instance, revealed_ids):
    """Build {req_id: Request} for the given revealed request IDs."""
    n_depots = int(instance['n_depots'])
    n_req    = int(instance['n_req'])
    tw       = instance['time_window'].squeeze(-1).numpy()

    out = {}
    for rid in revealed_ids:
        out[rid] = Request(
            req_id=rid,
            t_arrival=0.0,
            t_pickup=float(tw[n_depots + rid]),
            t_delivery=float(tw[n_depots + n_req + rid]),
            pickup_node=n_depots + rid,
            delivery_node=n_depots + n_req + rid,
            demand=float(instance['demand'][n_depots + rid].item()),
            status='waiting_for_pickup',
        )
    return out


def _build_solver():
    """Build a fresh (untrained) RLSolver by routing build_solver through a
    matched-dimension checkpoint we save on the fly."""
    import main as _m

    device = torch.device('cpu')
    # Dims here MUST match those build_solver constructs inside
    # main.py:build_solver — that function bakes hidden_node_dim=128 etc.
    actor = Model(
        input_node_dim=11, hidden_node_dim=128,
        input_edge_dim=4, hidden_edge_dim=16,
        conv_layers=3, arch='hetgat',
    ).to(device).eval()

    tmp = os.path.join(os.path.dirname(__file__), '_tmp_actor.pt')
    torch.save(actor.state_dict(), tmp)
    try:
        rl_solver = _m.build_solver(tmp, arch='hetgat')
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return rl_solver, rl_solver.actor, device


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mask_zeros_all_unrevealed_channels():
    """Unit-level: every expected tensor is zeroed for unrevealed nodes."""
    rng = np.random.default_rng(42)
    inst = create_instance(n_req=8, n_uav=2, n_adr=2,
                           n_depots_uav=1, n_depots_adr=1, rng=rng)
    inst['n_depots'] = 2
    inst['n_req']    = 8

    rl_solver, _, device = _build_solver()
    batch = rl_solver._batchify_instance(inst)

    # Reveal only requests {0, 2, 5}; the rest must be invisible
    revealed_ids = [0, 2, 5]
    active = _make_active_requests(inst, revealed_ids)

    masked = rl_solver._mask_unrevealed_nodes(batch, active, inst)

    n_depots = inst['n_depots']
    n_req    = inst['n_req']
    n_total  = n_depots + 2 * n_req

    revealed_nodes = set(range(n_depots))
    for rid in revealed_ids:
        revealed_nodes.add(n_depots + rid)
        revealed_nodes.add(n_depots + n_req + rid)
    unrevealed = sorted(set(range(n_total)) - revealed_nodes)

    # (1) node features zeroed
    assert torch.all(masked['x'][:, unrevealed, :] == 0.0)
    assert torch.all(masked['demand'][:, unrevealed] == 0.0)
    assert torch.all(masked['time_window'][:, unrevealed] == 0.0)

    # (2) adjacency masks zeroed on rows AND cols for unrevealed
    for key in ('mask_adjacency_uav', 'mask_adjacency_adr'):
        m = masked[key].view(1, n_total, n_total, -1)
        assert torch.all(m[:, unrevealed, :, :] == 0.0), f'{key} rows not zeroed'
        assert torch.all(m[:, :, unrevealed, :] == 0.0), f'{key} cols not zeroed'

    # (3) edge features zeroed on rows AND cols for unrevealed
    for key in ('edge_attr_uav', 'edge_attr_adr', 'edge_attr_d', 'edge_attr_r'):
        e = masked[key].view(1, n_total, n_total, -1)
        assert torch.all(e[:, unrevealed, :, :] == 0.0), f'{key} rows not zeroed'
        assert torch.all(e[:, :, unrevealed, :] == 0.0), f'{key} cols not zeroed'


def test_mask_preserves_revealed_values():
    """Revealed-node features must be unchanged after masking."""
    rng = np.random.default_rng(7)
    inst = create_instance(n_req=6, n_uav=2, n_adr=2,
                           n_depots_uav=1, n_depots_adr=1, rng=rng)
    inst['n_depots'] = 2
    inst['n_req']    = 6
    rl_solver, _, _ = _build_solver()
    batch = rl_solver._batchify_instance(inst)

    revealed_ids = [1, 3]
    active = _make_active_requests(inst, revealed_ids)

    n_depots, n_req = 2, 6
    revealed_nodes = list(range(n_depots))
    for rid in revealed_ids:
        revealed_nodes.append(n_depots + rid)
        revealed_nodes.append(n_depots + n_req + rid)

    masked = rl_solver._mask_unrevealed_nodes(batch, active, inst)

    assert torch.allclose(batch['x'][:, revealed_nodes, :],
                          masked['x'][:, revealed_nodes, :])
    assert torch.allclose(batch['demand'][:, revealed_nodes],
                          masked['demand'][:, revealed_nodes])


def test_mask_no_unrevealed_is_noop():
    """If every request is revealed, masking must return the input unchanged."""
    rng = np.random.default_rng(11)
    inst = create_instance(n_req=4, n_uav=2, n_adr=2,
                           n_depots_uav=1, n_depots_adr=1, rng=rng)
    inst['n_depots'] = 2
    inst['n_req']    = 4
    rl_solver, _, _ = _build_solver()
    batch = rl_solver._batchify_instance(inst)
    active = _make_active_requests(inst, list(range(4)))

    masked = rl_solver._mask_unrevealed_nodes(batch, active, inst)
    assert masked is batch, 'no-op path should return the same dict object'


def test_encoder_output_invariant_to_unrevealed_perturbation():
    """
    Gold-standard test: perturb unrevealed-node coords, demand, TWs and edge
    features ARBITRARILY.  After masking + encoder forward, the embeddings for
    revealed nodes must be bit-identical.  Any non-zero diff = information leak.
    """
    rng = np.random.default_rng(2025)
    inst_a = create_instance(n_req=8, n_uav=2, n_adr=2,
                             n_depots_uav=1, n_depots_adr=1, rng=rng)
    inst_a['n_depots'] = 2
    inst_a['n_req']    = 8

    # Deep-copy and stomp on every channel for unrevealed nodes BEFORE masking.
    revealed_ids = [0, 1, 4]
    n_depots, n_req = 2, 8
    n_total = n_depots + 2 * n_req
    revealed_nodes = set(range(n_depots))
    for rid in revealed_ids:
        revealed_nodes.add(n_depots + rid)
        revealed_nodes.add(n_depots + n_req + rid)
    unrevealed = sorted(set(range(n_total)) - revealed_nodes)

    inst_b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inst_a.items()}

    # Perturbation that would absolutely affect a non-masked encoder.
    rng_t = torch.Generator().manual_seed(99)
    inst_b['x'][unrevealed, :] = torch.randn(len(unrevealed), inst_b['x'].shape[-1],
                                              generator=rng_t)
    inst_b['demand'][unrevealed] = torch.randn(len(unrevealed), 1, generator=rng_t)
    inst_b['time_window'][unrevealed] = torch.randn(len(unrevealed), 1, generator=rng_t)

    # Perturb edge tensors on unrevealed rows AND cols
    for key in ('edge_attr_uav', 'edge_attr_adr', 'edge_attr_d', 'edge_attr_r',
                'mask_adjacency_uav', 'mask_adjacency_adr'):
        if key not in inst_b:
            continue
        v = inst_b[key]
        ch = v.shape[-1]
        v2d = v.view(n_total, n_total, ch).clone()
        v2d[unrevealed, :, :] = torch.randn(len(unrevealed), n_total, ch, generator=rng_t)
        v2d[:, unrevealed, :] = torch.randn(n_total, len(unrevealed), ch, generator=rng_t)
        inst_b[key] = v2d.view(n_total * n_total, ch)

    rl_solver, actor, _ = _build_solver()
    active = _make_active_requests(inst_a, revealed_ids)

    batch_a = rl_solver._mask_unrevealed_nodes(
        rl_solver._batchify_instance(inst_a), active, inst_a)
    batch_b = rl_solver._mask_unrevealed_nodes(
        rl_solver._batchify_instance(inst_b), active, inst_b)

    # All masked-out channels must match bitwise between batch_a and batch_b.
    for key in ('x', 'demand', 'time_window',
                'edge_attr_uav', 'edge_attr_adr',
                'edge_attr_d', 'edge_attr_r',
                'mask_adjacency_uav', 'mask_adjacency_adr'):
        if key not in batch_a:
            continue
        assert torch.equal(batch_a[key], batch_b[key]), (
            f'Channel "{key}" differs after masking — information leak from unrevealed nodes!'
        )

    # And the encoder forward output for revealed nodes must be identical.
    with torch.no_grad():
        h_uav_a, h_adr_a = actor.encoder(batch_a)
        h_uav_b, h_adr_b = actor.encoder(batch_b)

    rev = sorted(revealed_nodes)
    assert torch.allclose(h_uav_a[:, rev, :], h_uav_b[:, rev, :], atol=1e-6), (
        'UAV encoder leaks information from unrevealed nodes'
    )
    assert torch.allclose(h_adr_a[:, rev, :], h_adr_b[:, rev, :], atol=1e-6), (
        'ADR encoder leaks information from unrevealed nodes'
    )


def test_mask_handles_partial_reveal_via_solver_solve():
    """
    End-to-end: call RLSolver.solve on a residual where only half of the
    requests are revealed.  The solver must produce a valid route_plan with
    no legs touching unrevealed nodes.
    """
    rng = np.random.default_rng(3)
    # Instance with 1 UAV + 1 ADR so the fleet matches one-to-one
    inst = create_instance(n_req=6, n_uav=1, n_adr=1,
                           n_depots_uav=1, n_depots_adr=1, rng=rng)
    inst['n_depots'] = 2
    inst['n_req']    = 6

    rl_solver, _, _ = _build_solver()

    revealed_ids = [0, 2, 4]
    active = _make_active_requests(inst, revealed_ids)

    from dispatch_sim import Vehicle
    vehicles = {
        0: Vehicle(vehicle_id=0, mode='uav', current_node=0, current_time=0.0,
                   battery=6500.0, load=0.0, capacity=5.0),
        1: Vehicle(vehicle_id=1, mode='adr', current_node=1, current_time=0.0,
                   battery=4500.0, load=0.0, capacity=10.0),
    }
    residual = {
        'full_instance': inst,
        'active_requests': active,
        'vehicles': vehicles,
        'current_time': 0.0,
    }

    route_plan = rl_solver.solve(residual)
    assert isinstance(route_plan, dict)

    # The solver must NEVER produce a leg whose endpoints are unrevealed nodes.
    n_depots, n_req = 2, 6
    revealed_nodes = set(range(n_depots))
    for rid in revealed_ids:
        revealed_nodes.add(n_depots + rid)
        revealed_nodes.add(n_depots + n_req + rid)

    for vid, legs in route_plan.items():
        for leg in legs:
            assert int(leg.from_node) in revealed_nodes, (
                f'vehicle {vid} departs from unrevealed node {leg.from_node}')
            assert int(leg.to_node)   in revealed_nodes, (
                f'vehicle {vid} arrives at unrevealed node {leg.to_node}')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
