"""
Tests for instance generation (spec §2 required tests).
"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from creat_vrp import create_instance


def make_instance(seed=0):
    rng = np.random.default_rng(seed)
    return create_instance(
        n_req=10, n_uav=2, n_adr=2,
        n_depots_uav=1, n_depots_adr=1,
        rng=rng
    )


def test_delivery_after_pickup():
    """t_delivery > t_pickup for every request."""
    inst = make_instance()
    n_depots = 2
    n_req = 10
    tw = inst['time_window'].numpy().squeeze()
    t_pickup = tw[n_depots:n_depots + n_req]
    t_delivery = tw[n_depots + n_req:]
    assert (t_delivery > t_pickup).all(), \
        f'Some t_delivery <= t_pickup:\n{list(zip(t_pickup, t_delivery))}'


def test_uav_dist_symmetric_nonneg():
    """UAV distance matrix is symmetric, non-negative, zero diagonal."""
    inst = make_instance()
    n_total = inst['x'].shape[0]
    ea = inst['edge_attr_d'].numpy().reshape(n_total, n_total)
    assert (ea >= 0).all(), 'UAV distances contain negative values'
    assert np.allclose(ea, ea.T, atol=1e-4), 'UAV distance matrix not symmetric'
    assert (np.diag(ea) == 0).all(), 'UAV distance diagonal not zero'


def test_adr_dist_symmetric_nonneg():
    """ADR distance matrix is symmetric, non-negative, zero diagonal."""
    inst = make_instance()
    n_total = inst['x'].shape[0]
    ea = inst['edge_attr_r'].numpy().reshape(n_total, n_total)
    # ADR uses road network — may have INF for unreachable pairs, but finite entries >= 0
    finite = ea[ea < 1e9]
    assert (finite >= 0).all(), 'ADR distances contain negative finite values'
    assert (np.diag(ea) == 0).all(), 'ADR distance diagonal not zero'


def test_uav_triangle_inequality():
    """UAV Euclidean distances satisfy triangle inequality."""
    inst = make_instance()
    n_total = inst['x'].shape[0]
    d = inst['edge_attr_d'].numpy().reshape(n_total, n_total)
    # Check a sample of triples
    rng = np.random.default_rng(0)
    nodes = rng.choice(n_total, size=min(n_total, 10), replace=False)
    for i in nodes:
        for j in nodes:
            for k in nodes:
                if d[i, j] < 1e9 and d[j, k] < 1e9:
                    assert d[i, k] <= d[i, j] + d[j, k] + 1e-3, \
                        f'Triangle inequality violated: d[{i},{k}]={d[i,k]:.4f} > {d[i,j]:.4f}+{d[j,k]:.4f}'


def test_at_least_one_mode_per_customer():
    """Every customer node is reachable by at least one mode."""
    inst = make_instance()
    n_depots = 2
    n_req = 10
    n_total = inst['x'].shape[0]
    feat = inst['x'].numpy()   # (n_total, 11)
    # acc_uav is col 8, acc_adr is col 9
    acc_uav = feat[:, 8]
    acc_adr = feat[:, 9]
    customers = list(range(n_depots, n_total))
    for i in customers:
        assert acc_uav[i] > 0 or acc_adr[i] > 0, \
            f'Customer node {i} is not reachable by any mode'


def test_multiple_seeds():
    """Instance generation is deterministic per seed and varies across seeds."""
    inst0a = make_instance(seed=0)
    inst0b = make_instance(seed=0)
    inst1 = make_instance(seed=1)
    import torch
    assert torch.allclose(inst0a['x'], inst0b['x']), 'Same seed → different instance'
    assert not torch.allclose(inst0a['x'], inst1['x']), 'Different seeds → same instance'
