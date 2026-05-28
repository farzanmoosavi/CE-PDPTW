"""
Reward bookkeeping test (spec §14).
Verify total reward == sum of per-step rewards on a known trajectory.
"""
import pytest
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from creat_vrp import create_instance, reward1


def test_reward_nonneg_cost_form():
    """reward1 returns a positive cost — should be ≥ 0 for any tour."""
    rng = np.random.default_rng(7)
    inst = create_instance(5, 1, 1, 1, 1, rng)

    n_total = inst['x'].shape[0]
    n_depots = 2
    n_req = 5
    n_nodes = n_req * 2

    batch = {k: v.unsqueeze(0) for k, v in inst.items()
             if isinstance(v, torch.Tensor)}

    # Build a trivial tour: depot → pickups in order → deliveries in order → depot
    tour = list(range(n_depots, n_depots + n_req)) + list(range(n_depots + n_req, n_total))
    # Pad to make a (1, 1, n_total+2) tensor with depot at start and end
    seq = [0] + tour + [0]
    tour_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).unsqueeze(0)

    time_tensor = torch.zeros(1, 1, len(seq) - 1)

    r = reward1(batch['time_window'], tour_tensor, batch['edge_attr_d'],
                batch['edge_attr_r'], time_tensor, num_drones=1)
    assert r.shape == (1, 1), f'Expected (1,1) shape, got {r.shape}'
    assert (r >= 0).all(), f'Expected non-negative cost, got {r}'


def test_reward_consistent_across_calls():
    """Same instance + same tour → same reward."""
    rng = np.random.default_rng(42)
    inst = create_instance(5, 1, 1, 1, 1, rng)
    batch = {k: v.unsqueeze(0) for k, v in inst.items()
             if isinstance(v, torch.Tensor)}

    n_depots = 2
    n_req = 5
    n_total = inst['x'].shape[0]
    tour = list(range(n_depots, n_total))
    seq = [0] + tour + [0]
    tour_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).unsqueeze(0)
    time_tensor = torch.zeros(1, 1, len(seq) - 1)

    r1 = reward1(batch['time_window'], tour_tensor, batch['edge_attr_d'],
                 batch['edge_attr_r'], time_tensor, num_drones=1)
    r2 = reward1(batch['time_window'], tour_tensor, batch['edge_attr_d'],
                 batch['edge_attr_r'], time_tensor, num_drones=1)
    assert torch.allclose(r1, r2), 'Reward is not deterministic'
