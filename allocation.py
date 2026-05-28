from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np

def char_fn(phi_uav: float, phi_adr: float, phi_coalition: float) -> Dict[frozenset, float]:
    return {
        frozenset(): 0.0,
        frozenset({'uav'}): phi_uav,
        frozenset({'adr'}): phi_adr,
        frozenset({'uav', 'adr'}): phi_coalition,
    }

def shapley(phi_uav: float, phi_adr: float, phi_coalition: float) -> Tuple[float, float]:
    sh_uav = (phi_uav + phi_coalition - phi_adr) / 2.0
    sh_adr = (phi_adr + phi_coalition - phi_uav) / 2.0
    return sh_uav, sh_adr

def epm(phi_uav: float, phi_adr: float, phi_coalition: float) -> Tuple[float, float]:
    saving = phi_uav + phi_adr - phi_coalition
    equal_share = saving / 2.0
    c_uav = phi_uav - equal_share
    c_adr = phi_adr - equal_share
    return c_uav, c_adr

def pam(phi_uav: float, phi_adr: float, phi_coalition: float) -> Tuple[float, float]:
    total_alone = phi_uav + phi_adr
    if total_alone < 1e-12:
        return phi_coalition / 2.0, phi_coalition / 2.0
    c_uav = phi_coalition * phi_uav / total_alone
    c_adr = phi_coalition * phi_adr / total_alone
    return c_uav, c_adr

def core_exists(c_uav: float, c_adr: float, phi_uav: float, phi_adr: float) -> bool:
    return c_uav <= phi_uav and c_adr <= phi_adr

def efficient(c_uav: float, c_adr: float, phi_coalition: float, tol: float = 1e-8) -> bool:
    return abs((c_uav + c_adr) - phi_coalition) <= tol

def participation_flags(c_uav: float, c_adr: float, phi_uav: float, phi_adr: float) -> Dict[str, bool]:
    return {
        'uav': c_uav <= phi_uav,
        'adr': c_adr <= phi_adr,
    }

def participation_rate(allocations_uav, allocations_adr, standalone_uav, standalone_adr):
    au = np.array(allocations_uav)
    aa = np.array(allocations_adr)
    su = np.array(standalone_uav)
    sa = np.array(standalone_adr)
    rate_uav = float((au <= su).mean())
    rate_adr = float((aa <= sa).mean())
    return rate_uav, rate_adr

def allocate(result: dict) -> dict:
    phi_uav = result['phi_uav_mean']
    phi_adr = result['phi_adr_mean']
    phi_coal = result['phi_coalition_mean']

    sh_u, sh_a = shapley(phi_uav, phi_adr, phi_coal)
    ep_u, ep_a = epm(phi_uav, phi_adr, phi_coal)
    pa_u, pa_a = pam(phi_uav, phi_adr, phi_coal)

    return {
        **result,
        'shapley': {'uav': sh_u, 'adr': sh_a},
        'epm': {'uav': ep_u, 'adr': ep_a},
        'pam': {'uav': pa_u, 'adr': pa_a},
        'core_shapley': core_exists(sh_u, sh_a, phi_uav, phi_adr),
        'core_epm': core_exists(ep_u, ep_a, phi_uav, phi_adr),
        'core_pam': core_exists(pa_u, pa_a, phi_uav, phi_adr),
        'efficient_shapley': efficient(sh_u, sh_a, phi_coal),
        'efficient_epm': efficient(ep_u, ep_a, phi_coal),
        'efficient_pam': efficient(pa_u, pa_a, phi_coal),
        'participation_shapley': participation_flags(sh_u, sh_a, phi_uav, phi_adr),
        'participation_epm': participation_flags(ep_u, ep_a, phi_uav, phi_adr),
        'participation_pam': participation_flags(pa_u, pa_a, phi_uav, phi_adr),
    }

def per_episode_analysis(
    costs_uav: List[float],
    costs_adr: List[float],
    costs_coal: List[float],
) -> Dict[str, float]:
    u = np.array(costs_uav, dtype=float)
    a = np.array(costs_adr, dtype=float)
    c = np.array(costs_coal, dtype=float)
    n = len(u)
    if n == 0:
        return {}

    out: Dict[str, float] = {}
    for name, fn in [('shapley', shapley), ('epm', epm), ('pam', pam)]:
        alloc_u = np.zeros(n)
        alloc_a = np.zeros(n)
        for i in range(n):
            alloc_u[i], alloc_a[i] = fn(float(u[i]), float(a[i]), float(c[i]))

        uav_gain = u - alloc_u
        adr_gain = a - alloc_a
        uav_ir = uav_gain >= 0
        adr_ir = adr_gain >= 0
        core = uav_ir & adr_ir

        out[f'uav_ir_rate_{name}']   = float(uav_ir.mean())
        out[f'adr_ir_rate_{name}']   = float(adr_ir.mean())
        out[f'core_rate_{name}']     = float(core.mean())
        out[f'uav_gain_mean_{name}'] = float(uav_gain.mean())
        out[f'adr_gain_mean_{name}'] = float(adr_gain.mean())
        out[f'uav_gain_p5_{name}']   = float(np.percentile(uav_gain, 5))
        out[f'adr_gain_p5_{name}']   = float(np.percentile(adr_gain, 5))

    total_gain = u + a - c
    out['coalition_gain_mean']          = float(total_gain.mean())
    out['coalition_gain_std']           = float(total_gain.std())
    out['coalition_gain_p5']            = float(np.percentile(total_gain, 5))
    out['coalition_gain_positive_rate'] = float((total_gain >= 0).mean())
    return out

def conviction_score(result: Dict[str, float]) -> float:
    cr   = result.get('core_rate_shapley', 0.0)
    mg   = max(result.get('coalition_gain_mean', 0.0), 0.0)
    p5   = result.get('coalition_gain_p5', -1.0)

    magnitude_score  = float(np.tanh(mg / 10.0))
    downside_penalty = 1.0 if p5 >= 0.0 else max(0.0, 1.0 + p5 / 10.0)

    return float(cr * magnitude_score * downside_penalty)
