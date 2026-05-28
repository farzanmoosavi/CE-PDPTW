"""
Precedence test (spec §14).
Every delivery must be served by the same vehicle as its pickup, and after it.
"""
import pytest
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from vrpUpdate import update_mask, update_state


def test_delivery_blocked_before_pickup():
    """
    Delivery nodes should be masked (inaccessible) until the corresponding
    pickup has been visited by the same vehicle.
    """
    batch_size = 1
    n_uav = 1
    n_adr = 0
    n_agents = n_uav + n_adr if n_adr > 0 else 1
    num_depots = 1
    n_req = 3
    num_nodes = num_depots + n_req * 2

    # demand: [0, +1, +2, +3, -1, -2, -3]
    demand = torch.zeros(batch_size, num_nodes, 1)
    demand[0, 1, 0] = 1.0
    demand[0, 2, 0] = 2.0
    demand[0, 3, 0] = 3.0
    demand[0, 4, 0] = -1.0
    demand[0, 5, 0] = -2.0
    demand[0, 6, 0] = -3.0

    capacity = torch.ones(batch_size, n_agents, 1) * 5.0
    battery = torch.ones(batch_size, n_agents, 1) * 6500.0
    E = [6500.0, 0.0]

    mask1 = torch.zeros(batch_size, n_agents, num_nodes)

    # Initial selection: depot
    selected = torch.zeros(batch_size, n_agents, dtype=torch.long)
    mask, mask1 = update_mask(demand, capacity, selected, mask1, battery, n_uav, E, 0)

    # Delivery nodes (4,5,6) should be masked before their pickups are visited
    for d_node in [4, 5, 6]:
        assert mask[0, 0, d_node] == 1, \
            f'Delivery node {d_node} should be blocked before pickup, got mask={mask[0, 0, d_node]}'

    # Pickup nodes should be accessible
    for p_node in [1, 2, 3]:
        assert mask[0, 0, p_node] == 0, \
            f'Pickup node {p_node} should be accessible, got mask={mask[0, 0, p_node]}'


def test_delivery_unblocked_after_pickup():
    """After visiting pickup 1, delivery 4 should become accessible."""
    batch_size = 1
    n_uav = 1
    n_agents = 1
    num_depots = 1
    n_req = 2
    num_nodes = num_depots + n_req * 2

    demand = torch.zeros(batch_size, num_nodes, 1)
    demand[0, 1, 0] = 1.0
    demand[0, 2, 0] = 2.0
    demand[0, 3, 0] = -1.0
    demand[0, 4, 0] = -2.0

    capacity = torch.ones(batch_size, n_agents, 1) * 5.0
    battery = torch.ones(batch_size, n_agents, 1) * 6500.0
    E = [6500.0, 0.0]
    mask1 = torch.zeros(batch_size, n_agents, num_nodes)

    # Step 0: start at depot
    selected = torch.zeros(batch_size, n_agents, dtype=torch.long)
    mask, mask1 = update_mask(demand, capacity, selected, mask1, battery, n_uav, E, 0)

    # Step 1: visit pickup node 1
    selected = torch.ones(batch_size, n_agents, dtype=torch.long)  # node 1
    mask, mask1 = update_mask(demand, capacity, selected, mask1, battery, n_uav, E, 1)

    # Delivery of pickup 1 (node 3) should now be accessible
    assert mask[0, 0, 3] == 0, \
        f'Delivery node 3 should be accessible after visiting pickup 1, got mask={mask[0, 0, 3]}'
