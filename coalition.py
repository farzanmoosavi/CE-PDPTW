import numpy as np
from typing import Dict, List, Any
from dispatch_sim import RollingHorizonDispatcher, sample_arrival_stream, Vehicle
from creat_vrp import create_instance
from allocation import per_episode_analysis, conviction_score

def _episode_total_cost(episode_log: List[Dict]) -> float:
    summary = next((e for e in episode_log if e.get('summary')), {})
    return float(summary.get('total_cost', 0.0))

def _episode_undelivered(episode_log: List[Dict]) -> int:
    summary = next((e for e in episode_log if e.get('summary')), {})
    return int(summary.get('undelivered', 0))

def _episode_late_pct(episode_log: List[Dict]) -> float:
    summary = next((e for e in episode_log if e.get('summary')), {})
    return float(summary.get('tw_delivery_violation_rate', 0.0))

def _episode_served_frac(episode_log: List[Dict]) -> float:
    summary = next((e for e in episode_log if e.get('summary')), {})
    revealed = int(summary.get('total_revealed', 0))
    delivered = int(summary.get('total_delivered', 0))
    return delivered / max(revealed, 1)

def _assign_market(arrivals: List[Dict], uav_frac: float, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    return ['uav' if rng.random() < uav_frac else 'adr' for _ in arrivals]

def summarise(
    uav_vals: List[float],
    adr_vals: List[float],
    coalition_vals: List[float],
    revenue_uav: float = 0.0,
    revenue_adr: float = 0.0,
) -> Dict[str, Any]:
    u = np.array(uav_vals)
    a = np.array(adr_vals)
    c = np.array(coalition_vals)
    savings = u + a - c
    total_standalone = u.mean() + a.mean()
    uav_share = u.mean() / max(total_standalone, 1e-9)
    adr_share = a.mean() / max(total_standalone, 1e-9)
    sh_uav_cost = (u.mean() + c.mean() - a.mean()) / 2.0
    sh_adr_cost = (a.mean() + c.mean() - u.mean()) / 2.0
    return {
        'phi_uav_mean': float(u.mean()),
        'phi_uav_std': float(u.std()),
        'phi_adr_mean': float(a.mean()),
        'phi_adr_std': float(a.std()),
        'phi_coalition_mean': float(c.mean()),
        'phi_coalition_std': float(c.std()),
        'coalition_gain_mean': float(savings.mean()),
        'coalition_gain_std': float(savings.std()),
        'core_exists': bool((savings >= 0).mean() >= 0.5),
        'core_freq': float((savings >= 0).mean()),
        'uav_benefit_mean': float(u.mean() - sh_uav_cost),
        'adr_benefit_mean': float(a.mean() - sh_adr_cost),
        'uav_profit_standalone': float(revenue_uav - u.mean()),
        'adr_profit_standalone': float(revenue_adr - a.mean()),
        'uav_profit_coalition': float(revenue_uav - sh_uav_cost),
        'adr_profit_coalition': float(revenue_adr - sh_adr_cost),
    }

def make_fleet(
    n_uav: int,
    n_adr: int,
    n_depots_uav: int,
    n_depots_adr: int,
    battery_uav: float = 6500.0,
    battery_adr: float = 4500.0,
    q_uav: float = 5.0,
    q_adr: float = 10.0,
) -> Dict[int, Vehicle]:
    vehicles: Dict[int, Vehicle] = {}

    for i in range(n_uav):
        start_node = i % max(n_depots_uav, 1)
        vehicles[i] = Vehicle(
            vehicle_id=i,
            mode='uav',
            current_node=start_node,
            current_time=0.0,
            battery=battery_uav,
            load=0.0,
            capacity=q_uav,
            battery_init=battery_uav,
        )

    for i in range(n_adr):
        vid = n_uav + i
        start_node = n_depots_uav + (i % max(n_depots_adr, 1))
        vehicles[vid] = Vehicle(
            vehicle_id=vid,
            mode='adr',
            current_node=start_node,
            current_time=0.0,
            battery=battery_adr,
            load=0.0,
            capacity=q_adr,
            battery_init=battery_adr,
        )

    return vehicles

def coalition_cost_for_config(
    solver,
    n_episodes: int,
    market_split: Dict[str, float],
    eps_shared: float,
    depot_shared: bool,
    alpha_asymmetry: float,
    delta_minutes: float = 10.0,
    shift_minutes: float = 120.0,
    peak_rate: float = 20.0,
    n_uav: int = 3,
    n_adr: int = 3,
    n_req: int = 25,
    seed_base: int = 0,
    charging_cost_uav: float = 0.0,
    charging_cost_adr: float = 0.0,
    revenue_per_req_uav: float = 0.0,
    revenue_per_req_adr: float = 0.0,
    use_real_world: bool = False,
    networks=None,
) -> Dict[str, Any]:
    costs_uav, costs_adr, costs_coal = [], [], []
    violations = []
    late_uav, late_adr, late_coal = [], [], []
    served_uav, served_adr, served_coal = [], [], []
    uav_frac = market_split.get('uav', 0.5)

    if use_real_world:
        from mississauga_instance import (
            load_mississauga_networks,
            create_mississauga_instance,
        )
        _nets = networks if networks is not None else load_mississauga_networks()

    for ep in range(n_episodes):
        rng = np.random.default_rng(seed_base + ep)
        n_dep_uav = 2 if depot_shared else 1
        n_dep_adr = 2 if depot_shared else 1

        if use_real_world:
            instance = create_mississauga_instance(
                n_req=n_req,
                n_uav=n_uav,
                n_adr=n_adr,
                n_depots_uav=n_dep_uav,
                n_depots_adr=n_dep_adr,
                rng=rng,
                networks=_nets,
            )
        else:
            instance = create_instance(
                n_req=n_req,
                n_uav=n_uav,
                n_adr=n_adr,
                n_depots_uav=n_dep_uav,
                n_depots_adr=n_dep_adr,
                rng=rng,
            )

        arrivals = sample_arrival_stream(
            shift_minutes,
            peak_rate,
            seed=seed_base + ep,
            max_requests=int(instance['n_req']),
        )

        market_assign = _assign_market(arrivals, uav_frac, seed=seed_base + ep + 100_000)
        uav_arrivals = [a for a, m in zip(arrivals, market_assign) if m == 'uav']
        adr_arrivals = [a for a, m in zip(arrivals, market_assign) if m == 'adr']

        rng_share = np.random.default_rng(seed_base + ep + 200_000)
        uav_ext = uav_arrivals + [a for a in adr_arrivals if rng_share.random() < eps_shared]
        adr_ext = adr_arrivals + [a for a in uav_arrivals if rng_share.random() < eps_shared]
        uav_ext.sort(key=lambda x: x['t_arrival'])
        adr_ext.sort(key=lambda x: x['t_arrival'])

        base_fleet = make_fleet(
            n_uav=n_uav,
            n_adr=n_adr,
            n_depots_uav=n_dep_uav,
            n_depots_adr=n_dep_adr,
        )

        uav_fleet = {k: v for k, v in base_fleet.items() if v.mode == 'uav'}
        uav_log = RollingHorizonDispatcher(
            solver, delta_minutes, shift_minutes, alpha_p_ratio=alpha_asymmetry
        ).run_shift(uav_ext, uav_fleet, instance)
        phi_uav = _episode_total_cost(uav_log)

        adr_fleet = {k: v for k, v in base_fleet.items() if v.mode == 'adr'}
        adr_log = RollingHorizonDispatcher(
            solver, delta_minutes, shift_minutes, alpha_p_ratio=alpha_asymmetry
        ).run_shift(adr_ext, adr_fleet, instance)
        phi_adr = _episode_total_cost(adr_log)

        joint_fleet = make_fleet(
            n_uav=n_uav,
            n_adr=n_adr,
            n_depots_uav=n_dep_uav,
            n_depots_adr=n_dep_adr,
        )
        joint_log = RollingHorizonDispatcher(
            solver, delta_minutes, shift_minutes, alpha_p_ratio=alpha_asymmetry
        ).run_shift(arrivals, joint_fleet, instance)
        phi_coal = _episode_total_cost(joint_log)

        costs_uav.append(phi_uav + charging_cost_uav)
        costs_adr.append(phi_adr + charging_cost_adr)
        shared_infra_cost = (
            min(charging_cost_uav, charging_cost_adr)
            if depot_shared else charging_cost_uav + charging_cost_adr
        )
        costs_coal.append(phi_coal + shared_infra_cost)
        violations.append(_episode_undelivered(joint_log))

        late_uav.append(_episode_late_pct(uav_log))
        late_adr.append(_episode_late_pct(adr_log))
        late_coal.append(_episode_late_pct(joint_log))
        served_uav.append(_episode_served_frac(uav_log))
        served_adr.append(_episode_served_frac(adr_log))
        served_coal.append(_episode_served_frac(joint_log))

    uav_frac = market_split.get('uav', 0.5)
    revenue_uav_ep = revenue_per_req_uav * len(arrivals) * uav_frac
    revenue_adr_ep = revenue_per_req_adr * len(arrivals) * (1.0 - uav_frac)

    result = summarise(
        costs_uav, costs_adr, costs_coal,
        revenue_uav=revenue_uav_ep,
        revenue_adr=revenue_adr_ep,
    )
    result['mean_undelivered'] = float(np.mean(violations))
    result['n_req'] = n_req

    result['uav_late_pct_mean']    = float(np.mean(late_uav))
    result['adr_late_pct_mean']    = float(np.mean(late_adr))
    result['coal_late_pct_mean']   = float(np.mean(late_coal))
    result['uav_served_frac_mean'] = float(np.mean(served_uav))
    result['adr_served_frac_mean'] = float(np.mean(served_adr))
    result['coal_served_frac_mean']= float(np.mean(served_coal))
    result['uav_cost_per_req']     = result['phi_uav_mean']       / max(n_req, 1)
    result['adr_cost_per_req']     = result['phi_adr_mean']       / max(n_req, 1)
    result['coal_cost_per_req']    = result['phi_coalition_mean'] / max(n_req, 1)

    result['config'] = {
        'market_split': market_split,
        'eps_shared': eps_shared,
        'depot_shared': depot_shared,
        'alpha_asymmetry': alpha_asymmetry,
        'delta_minutes': delta_minutes,
        'charging_cost_uav': charging_cost_uav,
        'charging_cost_adr': charging_cost_adr,
    }

    ep_stability = per_episode_analysis(costs_uav, costs_adr, costs_coal)
    result.update(ep_stability)
    result['conviction'] = conviction_score(result)

    return result

def _run_one_config(args):
    (solver, n_episodes, ms, eps, ds, aa, cu, ca,
     delta_minutes, seed, n_req, n_uav, n_adr) = args
    return coalition_cost_for_config(
        solver=solver,
        n_episodes=n_episodes,
        market_split=ms,
        eps_shared=eps,
        depot_shared=ds,
        alpha_asymmetry=aa,
        delta_minutes=delta_minutes,
        seed_base=seed,
        charging_cost_uav=cu,
        charging_cost_adr=ca,
        n_req=n_req,
        n_uav=n_uav,
        n_adr=n_adr,
    )

SWEEP_AXES = {
    'market_split': [
        {'uav': 0.30, 'adr': 0.70},
        {'uav': 0.50, 'adr': 0.50},
        {'uav': 0.70, 'adr': 0.30},
    ],
    'eps_shared': [0.25, 0.50],
    'depot_shared': [False, True],
    'alpha_asymmetry': [1.0, 0.67, 0.50],
    'charging_infra': [
        (0.0, 0.0),
        (5.0, 2.0),
        (2.0, 5.0),
    ],
}

def run_full_sweep(
    solver,
    n_episodes: int = 200,
    delta_minutes: float = 10.0,
    seed_base: int = 0,
    n_workers: int = 1,
    n_req: int = 25,
    n_uav: int = 5,
    n_adr: int = 4,
) -> List[Dict]:
    from itertools import product as _product

    combos = list(_product(
        SWEEP_AXES['market_split'],
        SWEEP_AXES['eps_shared'],
        SWEEP_AXES['depot_shared'],
        SWEEP_AXES['alpha_asymmetry'],
        SWEEP_AXES['charging_infra'],
    ))

    print(
        f'Running sweep: {len(combos)} configs × {n_episodes} episodes = '
        f'{len(combos) * n_episodes} dispatcher runs'
        + (f'  [{n_workers} workers]' if n_workers > 1 else '')
    )

    task_args = [
        (solver, n_episodes, ms, eps, ds, aa, cu, ca,
         delta_minutes, seed_base + i * n_episodes, n_req, n_uav, n_adr)
        for i, (ms, eps, ds, aa, (cu, ca)) in enumerate(combos)
    ]

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_run_one_config, task_args))
    else:
        results = []
        for i, args in enumerate(task_args):
            ms, eps, ds, aa, cu, ca = args[2], args[3], args[4], args[5], args[6], args[7]
            print(
                f'Config {i + 1}/{len(combos)}: '
                f'split={ms}, eps={eps}, depot_shared={ds}, alpha_asym={aa}, '
                f'charging=({cu},{ca})'
            )
            results.append(_run_one_config(args))

    return results
