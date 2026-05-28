"""
Battery monotonicity test (spec §14).
Battery must be monotone non-increasing between depot visits.
"""
import pytest
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from vrpUpdate import update_state


def test_battery_decreases_on_travel():
    """Battery after a travel leg should be ≤ battery before."""
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

    time_window = torch.zeros(batch_size, num_nodes)
    time_window[0, 1] = 10.0
    time_window[0, 2] = 20.0

    BATTERY_INIT = 6500.0
    battery = torch.ones(batch_size, n_agents, 1) * BATTERY_INIT
    T_t = torch.zeros(batch_size, n_agents, 1)
    capacity = torch.ones(batch_size, n_agents, 1) * 5.0
    E = [BATTERY_INIT, 0.0]

    # Edge attr: distance 1.0 between node 0 and node 1
    edge_attr_uav = torch.ones(batch_size, num_nodes, num_nodes) * 0.5
    torch.diagonal(edge_attr_uav.view(num_nodes, num_nodes)).fill_(0.0)

    actions = [
        torch.zeros(batch_size, n_agents, 1, dtype=torch.long),   # from depot (0)
        torch.ones(batch_size, n_agents, 1, dtype=torch.long),    # to pickup (1)
    ]

    new_cap, new_time, new_battery = update_state(
        demand, time_window, battery, T_t, capacity, E,
        n_uav, actions, edge_attr_uav, edge_attr_uav
    )
    assert (new_battery <= BATTERY_INIT).all(), \
        f'Battery increased on travel: {BATTERY_INIT} → {new_battery}'


def test_battery_resets_at_depot():
    """Battery resets to E[0] when vehicle returns to depot."""
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

    time_window = torch.zeros(batch_size, num_nodes)
    BATTERY_INIT = 6500.0
    LOW_BATTERY = 1000.0
    battery = torch.ones(batch_size, n_agents, 1) * LOW_BATTERY
    T_t = torch.zeros(batch_size, n_agents, 1)
    capacity = torch.ones(batch_size, n_agents, 1) * 5.0
    E = [BATTERY_INIT, 0.0]

    edge_attr_uav = torch.ones(batch_size, num_nodes, num_nodes) * 0.5
    torch.diagonal(edge_attr_uav.view(num_nodes, num_nodes)).fill_(0.0)

    # From pickup (1) back to depot (0)
    actions = [
        torch.ones(batch_size, n_agents, 1, dtype=torch.long),    # from pickup (1)
        torch.zeros(batch_size, n_agents, 1, dtype=torch.long),   # to depot (0)
    ]
    _, _, new_battery = update_state(
        demand, time_window, battery, T_t, capacity, E,
        n_uav, actions, edge_attr_uav, edge_attr_uav
    )
    assert torch.isclose(new_battery, torch.tensor(BATTERY_INIT), atol=1.0), \
        f'Battery not reset at depot: {new_battery}'


def test_battery_nonneg():
    """Battery must never go negative."""
    batch_size = 1
    n_uav = 1
    n_agents = 1
    num_depots = 1
    n_req = 2
    num_nodes = num_depots + n_req * 2

    demand = torch.zeros(batch_size, num_nodes, 1)
    demand[0, 1, 0] = 1.0
    time_window = torch.zeros(batch_size, num_nodes)
    battery = torch.ones(batch_size, n_agents, 1) * 1.0   # nearly empty
    T_t = torch.zeros(batch_size, n_agents, 1)
    capacity = torch.ones(batch_size, n_agents, 1) * 5.0
    E = [6500.0, 0.0]

    edge_attr_uav = torch.ones(batch_size, num_nodes, num_nodes) * 2.0
    torch.diagonal(edge_attr_uav.view(num_nodes, num_nodes)).fill_(0.0)

    actions = [
        torch.zeros(batch_size, n_agents, 1, dtype=torch.long),
        torch.ones(batch_size, n_agents, 1, dtype=torch.long),
    ]
    _, _, new_battery = update_state(
        demand, time_window, battery, T_t, capacity, E,
        n_uav, actions, edge_attr_uav, edge_attr_uav
    )
    assert (new_battery >= 0).all(), f'Battery went negative: {new_battery}'
