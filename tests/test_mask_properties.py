"""
Unit tests for update_mask / update_state properties (vrpUpdate.py).

Node layout: [0..num_depots-1 | num_depots..+n_pickup-1 | num_depots+n_pickup..]
             depots              pickups                    deliveries
"""

import pytest
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vrpUpdate import update_mask, update_state


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_tensors(
    batch_size=1, num_depots=1, n_pickup=3,
    num_uav=1, num_adr=1,
    demand_val=1.0, capacity_val=5.0,
    battery_uav=6500.0, battery_adr=4500.0,
):
    n_total  = num_depots + 2 * n_pickup
    n_agents = num_uav + num_adr

    demand = torch.zeros(batch_size, n_total)
    demand[:, num_depots:num_depots + n_pickup]   =  demand_val
    demand[:, num_depots + n_pickup:]              = -demand_val  # delivery restores cap

    capacity = torch.full((batch_size, n_agents, 1), capacity_val)
    battery  = torch.zeros(batch_size, n_agents, 1)
    battery[:, :num_uav]  = battery_uav
    battery[:, num_uav:]  = battery_adr

    E = [battery_uav, battery_adr]
    return demand, capacity, battery, E, n_total, n_agents


def _zero_mask(batch_size, n_agents, n_total):
    return torch.zeros(batch_size, n_agents, n_total)


def _at_depot(batch_size, n_agents):
    return torch.zeros(batch_size, n_agents, dtype=torch.long)


def _small_edge_matrix(batch_size, n_total, inf_pairs=()):
    """Finite-distance edge matrix with optional INF pairs."""
    d = torch.ones(batch_size, n_total, n_total) * 0.5
    d[:, range(n_total), range(n_total)] = 0.0
    for (i, j) in inf_pairs:
        d[:, i, j] = 1e10
    return d


# ---------------------------------------------------------------------------
# 1. UAV agents cannot visit UAV-inaccessible nodes
# ---------------------------------------------------------------------------

def test_uav_inaccessible_blocks_uav_only():
    B, num_uav, num_adr = 1, 1, 1
    demand, capacity, battery, E, n_total, n_agents = _make_tensors(
        batch_size=B, num_uav=num_uav, num_adr=num_adr
    )
    # acc_uav=0 for all customer nodes → UAV blocked; ADR unaffected
    acc_uav = torch.zeros(B, n_total)
    acc_uav[:, 0] = 1.0   # depot stays accessible
    acc_adr = torch.ones(B, n_total)

    final_mask, _ = update_mask(
        demand, capacity, _at_depot(B, n_agents), _zero_mask(B, n_agents, n_total),
        battery, num_uav, E, i=0, acc_uav=acc_uav, acc_adr=acc_adr,
    )

    assert final_mask[0, 0, 1:].all(), \
        "UAV must not visit any UAV-inaccessible customer node"
    assert not final_mask[0, 1, 1:].all(), \
        "ADR must still be able to visit accessible customer nodes"


# ---------------------------------------------------------------------------
# 2. ADR agents cannot visit ADR-inaccessible nodes
# ---------------------------------------------------------------------------

def test_adr_inaccessible_blocks_adr_only():
    B, num_uav, num_adr = 1, 1, 1
    demand, capacity, battery, E, n_total, n_agents = _make_tensors(
        batch_size=B, num_uav=num_uav, num_adr=num_adr
    )
    acc_uav = torch.ones(B, n_total)
    acc_adr = torch.zeros(B, n_total)
    acc_adr[:, 0] = 1.0   # depot stays accessible

    final_mask, _ = update_mask(
        demand, capacity, _at_depot(B, n_agents), _zero_mask(B, n_agents, n_total),
        battery, num_uav, E, i=0, acc_uav=acc_uav, acc_adr=acc_adr,
    )

    assert final_mask[0, 1, 1:].all(), \
        "ADR must not visit any ADR-inaccessible customer node"
    assert not final_mask[0, 0, 1:].all(), \
        "UAV must still be able to visit accessible customer nodes"


# ---------------------------------------------------------------------------
# 3. Depot nodes are never blocked by accessibility masks
# ---------------------------------------------------------------------------

def test_depot_accessible_despite_zero_acc():
    B, num_uav, num_adr = 1, 1, 1
    demand, capacity, battery, E, n_total, n_agents = _make_tensors(
        batch_size=B, num_uav=num_uav, num_adr=num_adr
    )
    # Block everything for both modes — depot override must still kick in
    acc_uav = torch.zeros(B, n_total)
    acc_adr = torch.zeros(B, n_total)

    final_mask, _ = update_mask(
        demand, capacity, _at_depot(B, n_agents), _zero_mask(B, n_agents, n_total),
        battery, num_uav, E, i=0, acc_uav=acc_uav, acc_adr=acc_adr,
    )

    assert final_mask[0, 0, 0].item() == 0, \
        "Depot must remain unmasked for UAV even when acc_uav=0 everywhere"
    assert final_mask[0, 1, 0].item() == 0, \
        "Depot must remain unmasked for ADR even when acc_adr=0 everywhere"


# ---------------------------------------------------------------------------
# 4. No double-serving — visited nodes stay masked in subsequent steps
# ---------------------------------------------------------------------------

def test_no_double_serving():
    B, num_uav, num_adr = 1, 1, 1
    num_depots = 1
    demand, capacity, battery, E, n_total, n_agents = _make_tensors(
        batch_size=B, num_depots=num_depots, num_uav=num_uav, num_adr=num_adr
    )

    mask0 = _zero_mask(B, n_agents, n_total)
    _, mask2_step0 = update_mask(
        demand, capacity, _at_depot(B, n_agents), mask0, battery, num_uav, E, i=0
    )

    # UAV visits pickup 1 (index 1)
    sel1 = torch.tensor([[1, 0]])
    final_mask, _ = update_mask(
        demand, capacity, sel1, mask2_step0, battery, num_uav, E, i=1
    )

    assert final_mask[0, 0, 1].item() != 0, \
        "Pickup node already visited by UAV must be masked in next step"


# ---------------------------------------------------------------------------
# 5. Delivery nodes blocked until paired pickup is done
# ---------------------------------------------------------------------------

def test_deliveries_blocked_at_start():
    B, num_uav, num_adr = 1, 1, 1
    num_depots, n_pickup = 1, 3
    demand, capacity, battery, E, n_total, n_agents = _make_tensors(
        batch_size=B, num_depots=num_depots, n_pickup=n_pickup,
        num_uav=num_uav, num_adr=num_adr
    )
    final_mask, _ = update_mask(
        demand, capacity, _at_depot(B, n_agents), _zero_mask(B, n_agents, n_total),
        battery, num_uav, E, i=0,
    )

    delivery_start = num_depots + n_pickup  # = 4
    assert final_mask[0, 0, delivery_start:].all(), \
        "All delivery nodes must be blocked before any pickup"
    assert final_mask[0, 1, delivery_start:].all(), \
        "All delivery nodes must be blocked for ADR before any pickup"


# ---------------------------------------------------------------------------
# 6. Delivery unblocked for the vehicle that performed the paired pickup
# ---------------------------------------------------------------------------

def test_delivery_unblocked_for_picker_only():
    B, num_uav, num_adr = 1, 1, 1
    num_depots, n_pickup = 1, 3
    demand, capacity, battery, E, n_total, n_agents = _make_tensors(
        batch_size=B, num_depots=num_depots, n_pickup=n_pickup,
        num_uav=num_uav, num_adr=num_adr
    )

    # Step 0: start at depot
    mask0 = _zero_mask(B, n_agents, n_total)
    _, mask2_step0 = update_mask(
        demand, capacity, _at_depot(B, n_agents), mask0, battery, num_uav, E, i=0
    )

    # Step 1: UAV picks up request 0 (node index num_depots = 1); ADR stays at depot
    sel1 = torch.tensor([[num_depots, 0]])
    final_mask, _ = update_mask(
        demand, capacity, sel1, mask2_step0, battery, num_uav, E, i=1
    )

    # Delivery of request 0 = num_depots + n_pickup + 0 = 4
    delivery0 = num_depots + n_pickup
    assert final_mask[0, 0, delivery0].item() == 0, \
        "Delivery node must be unblocked for the UAV that did the pickup"
    assert final_mask[0, 1, delivery0].item() != 0, \
        "Delivery node must stay blocked for ADR that did not do the pickup"


# ---------------------------------------------------------------------------
# 7. Low battery forces depot return for UAV
# ---------------------------------------------------------------------------

def test_low_battery_forces_uav_to_depot():
    B, num_uav, num_adr = 1, 1, 1
    num_depots, n_pickup = 1, 3
    demand, capacity, battery, E, n_total, n_agents = _make_tensors(
        batch_size=B, num_depots=num_depots, n_pickup=n_pickup,
        num_uav=num_uav, num_adr=num_adr, battery_uav=6500.0
    )
    battery_low = battery.clone()
    battery_low[0, 0, 0] = 6500.0 * 0.25 - 1.0   # just below 25 % threshold

    final_mask, _ = update_mask(
        demand, capacity, _at_depot(B, n_agents), _zero_mask(B, n_agents, n_total),
        battery_low, num_uav, E, i=0,
    )

    # All non-depot nodes must be blocked for the low-battery UAV
    assert final_mask[0, 0, num_depots:].all(), \
        "Low-battery UAV must be forced toward depot only"
    assert final_mask[0, 0, 0].item() == 0, \
        "Depot must remain accessible for low-battery UAV"


# ---------------------------------------------------------------------------
# 8. Low battery forces depot return for ADR
# ---------------------------------------------------------------------------

def test_low_battery_forces_adr_to_depot():
    B, num_uav, num_adr = 1, 1, 1
    num_depots, n_pickup = 1, 3
    demand, capacity, battery, E, n_total, n_agents = _make_tensors(
        batch_size=B, num_depots=num_depots, n_pickup=n_pickup,
        num_uav=num_uav, num_adr=num_adr, battery_adr=4500.0
    )
    battery_low = battery.clone()
    battery_low[0, 1, 0] = 4500.0 * 0.20 - 1.0   # just below 20 % threshold

    final_mask, _ = update_mask(
        demand, capacity, _at_depot(B, n_agents), _zero_mask(B, n_agents, n_total),
        battery_low, num_uav, E, i=0,
    )

    assert final_mask[0, 1, num_depots:].all(), \
        "Low-battery ADR must be forced toward depot only"
    assert final_mask[0, 1, 0].item() == 0, \
        "Depot must remain accessible for low-battery ADR"


# ---------------------------------------------------------------------------
# 9. Depot visit restores battery to full (update_state)
# ---------------------------------------------------------------------------

def test_depot_visit_restores_battery():
    B, num_uav, num_adr = 1, 1, 1
    num_depots, n_pickup = 1, 3
    demand, capacity, battery, E, n_total, n_agents = _make_tensors(
        batch_size=B, num_depots=num_depots, n_pickup=n_pickup,
        num_uav=num_uav, num_adr=num_adr, battery_uav=6500.0, battery_adr=4500.0
    )
    time_window = torch.zeros(B, n_total)
    T_t         = torch.zeros(B, n_agents, 1)

    edge_d = _small_edge_matrix(B, n_total)
    edge_r = _small_edge_matrix(B, n_total)

    # Drain both agents
    battery_low = battery.clone()
    battery_low[0, 0, 0] = 50.0   # UAV almost empty
    battery_low[0, 1, 0] = 50.0   # ADR almost empty

    # Previous action: agents were at pickup node 1; now they return to depot 0
    prev = torch.ones(B, n_agents, 1, dtype=torch.long)   # from pickup1
    curr = torch.zeros(B, n_agents, 1, dtype=torch.long)  # to depot0

    _, _, new_battery = update_state(
        demand, time_window, battery_low, T_t, capacity, E,
        num_uav, actions=[prev, curr],
        edge_attr_uav=edge_d, edge_attr_adr=edge_r,
    )

    assert new_battery[0, 0].item() == pytest.approx(E[0], abs=1.0), \
        "UAV battery must be fully restored after depot visit"
    assert new_battery[0, 1].item() == pytest.approx(E[1], abs=1.0), \
        "ADR battery must be fully restored after depot visit"


# ---------------------------------------------------------------------------
# 10. INF-distance arcs are blocked by update_mask
# ---------------------------------------------------------------------------

def test_inf_arc_blocked_for_uav():
    B, num_uav, num_adr = 1, 1, 1
    num_depots, n_pickup = 1, 3
    demand, capacity, battery, E, n_total, n_agents = _make_tensors(
        batch_size=B, num_depots=num_depots, n_pickup=n_pickup,
        num_uav=num_uav, num_adr=num_adr
    )
    # UAV at depot (0): arcs to nodes 2 and 3 are INF; arc to node 1 is finite
    edge_d = _small_edge_matrix(B, n_total, inf_pairs=[(0, 2), (0, 3)])
    edge_r = _small_edge_matrix(B, n_total)

    final_mask, _ = update_mask(
        demand, capacity, _at_depot(B, n_agents), _zero_mask(B, n_agents, n_total),
        battery, num_uav, E, i=0,
        acc_uav=torch.ones(B, n_total), acc_adr=torch.ones(B, n_total),
        edge_attr_d=edge_d, edge_attr_r=edge_r,
    )

    assert final_mask[0, 0, 2].item() != 0, \
        "Node 2 (INF distance from depot) must be blocked for UAV"
    assert final_mask[0, 0, 3].item() != 0, \
        "Node 3 (INF distance from depot) must be blocked for UAV"
    assert final_mask[0, 0, 1].item() == 0, \
        "Node 1 (finite distance from depot) must NOT be INF-blocked for UAV"


# ---------------------------------------------------------------------------
# 11. Capacity-exceeding pickup nodes are masked
# ---------------------------------------------------------------------------

def test_capacity_exceeded_masks_pickup():
    B, num_uav, num_adr = 1, 1, 1
    num_depots, n_pickup = 1, 3
    demand, capacity, battery, E, n_total, n_agents = _make_tensors(
        batch_size=B, num_depots=num_depots, n_pickup=n_pickup,
        num_uav=num_uav, num_adr=num_adr,
        demand_val=3.0, capacity_val=5.0
    )
    # Reduce UAV capacity to 2 — less than any pickup demand (3.0)
    capacity_low = capacity.clone()
    capacity_low[0, 0, 0] = 2.0

    final_mask, _ = update_mask(
        demand, capacity_low, _at_depot(B, n_agents), _zero_mask(B, n_agents, n_total),
        battery, num_uav, E, i=0,
    )

    for pn in range(num_depots, num_depots + n_pickup):
        assert final_mask[0, 0, pn].item() != 0, \
            f"Pickup node {pn} (demand 3 > capacity 2) must be masked for UAV"
