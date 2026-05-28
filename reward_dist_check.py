import argparse
import numpy as np
import torch

from creat_vrp import (create_instance, reward1,
                       V_UAV_COORD_PER_MIN, V_ADR_COORD_PER_MIN,
                       UAV_LAND_TAKEOFF_MIN)

RUNGS = {
    'A': dict(n_req=5,  n_uav=2,  n_adr=2, n_depots_uav=1, n_depots_adr=1),
    'B': dict(n_req=10, n_uav=4,  n_adr=3, n_depots_uav=1, n_depots_adr=1),
    'C': dict(n_req=25, n_uav=5,  n_adr=4, n_depots_uav=2, n_depots_adr=2),
}

Q_UAV = 5.0

def _build_tour(strategy: str, n_req: int, n_uav: int, n_adr: int,
                n_depots_uav: int, n_depots_adr: int,
                demand: np.ndarray) -> torch.Tensor:
    n_depots = n_depots_uav + n_depots_adr
    n_agent  = n_uav + n_adr
    steps    = n_depots + n_req * 2

    pickups_per_agent   = [[] for _ in range(n_agent)]
    deliveries_per_agent = [[] for _ in range(n_agent)]
    home = list(range(n_depots_uav)) * n_uav + list(range(n_depots_uav, n_depots)) * n_adr
    home = [home[i % len(home)] for i in range(n_agent)]

    demand_vals = demand[n_depots:n_depots + n_req]

    uav_ptr = 0
    adr_ptr = 0

    for r in range(n_req):
        pickup   = n_depots + r
        delivery = n_depots + n_req + r
        d        = float(demand_vals[r])

        if strategy in ('round_robin', 'batched'):
            if d <= Q_UAV:
                agent = uav_ptr % n_uav
                uav_ptr += 1
            else:
                agent = n_uav + (adr_ptr % n_adr)
                adr_ptr += 1
        elif strategy == 'uav_only':
            if d <= Q_UAV:
                agent = uav_ptr % n_uav
                uav_ptr += 1
            else:
                continue
        elif strategy == 'adr_only':
            agent = n_uav + (adr_ptr % n_adr)
            adr_ptr += 1
        else:
            raise ValueError(strategy)

        pickups_per_agent[agent].append(pickup)
        deliveries_per_agent[agent].append(delivery)

    tour = torch.zeros(1, n_agent, steps, dtype=torch.long)
    for ag in range(n_agent):
        if strategy == 'batched':
            seq = [home[ag]] + pickups_per_agent[ag] + deliveries_per_agent[ag]
        else:
            pairs = []
            for p, d_ in zip(pickups_per_agent[ag], deliveries_per_agent[ag]):
                pairs.extend([p, d_])
            seq = [home[ag]] + pairs
        while len(seq) < steps:
            seq.append(home[ag])
        tour[0, ag, :] = torch.tensor(seq[:steps], dtype=torch.long)

    return tour

def _compute_time_tensor(tour: torch.Tensor, inst_batch: dict,
                         n_uav: int, n_adr: int, n_depots: int, n_req: int
                         ) -> torch.Tensor:
    B, n_agent, steps = tour.shape
    n_total = n_depots + n_req * 2

    d_uav = inst_batch['edge_attr_d'].squeeze(-1).view(B, n_total, n_total)
    d_adr = inst_batch['edge_attr_r'].squeeze(-1).view(B, n_total, n_total)

    tw = inst_batch['time_window'].squeeze(-1)

    time_tensor = torch.zeros(B, n_agent, steps - 1)

    for b in range(B):
        for a in range(n_agent):
            is_uav = (a < n_uav)
            d_mat  = d_uav[b] if is_uav else d_adr[b]
            v      = V_UAV_COORD_PER_MIN if is_uav else V_ADR_COORD_PER_MIN

            t = 0.0
            for k in range(steps - 1):
                src = int(tour[b, a, k])
                dst = int(tour[b, a, k + 1])
                d   = float(d_mat[src, dst])
                if d >= 1e9:
                    d = 0.0
                t += d / v
                if n_depots <= dst < n_depots + n_req:
                    t_ready = float(tw[b, dst])
                    if t < t_ready:
                        t = t_ready
                if is_uav and dst >= n_depots:
                    t += UAV_LAND_TAKEOFF_MIN
                time_tensor[b, a, k] = t

    return time_tensor

def _eval_seed(seed: int, n_instances: int, cfg: dict,
               tw_slack_mean: float = 30.0, tw_slack_std: float = 5.0,
               tw_slack_clip: tuple = (15.0, 60.0),
               strategies=('round_robin', 'uav_only', 'adr_only')):
    n_req        = cfg['n_req']
    n_uav        = cfg['n_uav']
    n_adr        = cfg['n_adr']
    n_depots_uav = cfg['n_depots_uav']
    n_depots_adr = cfg['n_depots_adr']
    n_depots     = n_depots_uav + n_depots_adr
    n_agent      = n_uav + n_adr

    results = {s: {'total': [], 'travel_uav': [], 'travel_adr': [],
                   'tw_uav': [], 'tw_adr': [], 'undeliv': [],
                   'heavy_frac': []}
               for s in strategies}

    for i in range(n_instances):
        rng  = np.random.default_rng(seed + i)
        inst = create_instance(
            n_req, n_uav, n_adr, n_depots_uav, n_depots_adr, rng,
            tw_slack_mean=tw_slack_mean,
            tw_slack_std=tw_slack_std,
            tw_slack_clip=tw_slack_clip,
        )
        demand_np  = inst['demand'].squeeze(-1).numpy()
        heavy_frac = float(np.mean(demand_np[n_depots:n_depots + n_req] > Q_UAV))

        batch = {k: v.unsqueeze(0) for k, v in inst.items()
                 if isinstance(v, torch.Tensor)}

        for strat in strategies:
            tour = _build_tour(
                strat, n_req, n_uav, n_adr, n_depots_uav, n_depots_adr, demand_np
            )

            time_t = _compute_time_tensor(tour, batch, n_uav, n_adr, n_depots, n_req)

            with torch.no_grad():
                cost, comps = reward1(
                    batch['time_window'], tour,
                    batch['edge_attr_d'], batch['edge_attr_r'],
                    time_t, n_uav,
                    num_depots=n_depots,
                    return_breakdown=True,
                )

            total_per_req = float(cost[0].sum() / n_req)
            travel_uav    = float(comps['travel'][0, :n_uav].sum() / n_req)
            travel_adr    = float(comps['travel'][0, n_uav:].sum() / n_req)
            tw_uav        = float(comps['tw_penalty'][0, :n_uav].sum() / n_req)
            tw_adr        = float(comps['tw_penalty'][0, n_uav:].sum() / n_req)
            undeliv       = float(comps['undeliv'][0].sum() / n_req)

            r = results[strat]
            r['total'].append(total_per_req)
            r['travel_uav'].append(travel_uav)
            r['travel_adr'].append(travel_adr)
            r['tw_uav'].append(tw_uav)
            r['tw_adr'].append(tw_adr)
            r['undeliv'].append(undeliv)
            r['heavy_frac'].append(heavy_frac)

    return results

def _arr(lst):
    return np.array(lst, dtype=np.float64)

def _summarise(label: str, r: dict):
    total  = _arr(r['total'])
    tu     = _arr(r['travel_uav'])
    ta     = _arr(r['travel_adr'])
    twu    = _arr(r['tw_uav'])
    twa    = _arr(r['tw_adr'])
    ud     = _arr(r['undeliv'])
    hf     = _arr(r['heavy_frac'])
    print(f'  {label}')
    print(f'    total cost/req : {total.mean():.4f}  ±{total.std():.4f}')
    print(f'    travel UAV     : {tu.mean():.4f}  ±{tu.std():.4f}')
    print(f'    travel ADR     : {ta.mean():.4f}  ±{ta.std():.4f}')
    print(f'    TW penalty UAV : {twu.mean():.4f}  ±{twu.std():.4f}')
    print(f'    TW penalty ADR : {twa.mean():.4f}  ±{twa.std():.4f}')
    print(f'    undeliv cost   : {ud.mean():.4f}  ±{ud.std():.4f}')
    print(f'    heavy req frac : {hf.mean():.1%}  (demand > Q_UAV={Q_UAV})')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rung', default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--n',    type=int, default=100,
                        help='Instances per seed (default 100)')
    args = parser.parse_args()

    cfg   = RUNGS[args.rung]
    seeds = [1000, 1001, 1002]

    print('=' * 65)
    print(f'Reward Distribution Check  |  Rung {args.rung}  |  {args.n} inst/seed')
    print(f'n_req={cfg["n_req"]}  n_uav={cfg["n_uav"]}  n_adr={cfg["n_adr"]}  '
          f'Q_UAV={Q_UAV}  demand=[1,10]')
    print('=' * 65)

    print('\n>>> Normal windows (tw_slack_mean=30, clip=[15,60])')
    seed_totals = {s: {} for s in ['round_robin', 'uav_only', 'adr_only']}

    for seed in seeds:
        print(f'\n  Seed {seed}:')
        res = _eval_seed(seed, args.n, cfg)
        for strat in ('round_robin', 'uav_only', 'adr_only'):
            _summarise(strat, res[strat])
            seed_totals[strat][seed] = _arr(res[strat]['total'])

    print('\n  Cross-seed distinguishability (round_robin, normal windows):')
    means = [seed_totals['round_robin'][s].mean() for s in seeds]
    stds  = [seed_totals['round_robin'][s].std()  for s in seeds]
    spread = max(means) - min(means)
    pooled_std = float(np.mean(stds))
    snr = spread / (pooled_std + 1e-9)
    print(f'    Seed means: {[f"{m:.4f}" for m in means]}')
    print(f'    Spread (max-min): {spread:.4f}   Pooled std: {pooled_std:.4f}')
    if snr >= 0.3:
        print(f'    [PASS] Seeds are distinguishable (SNR={snr:.2f} >= 0.3)')
    else:
        print(f'    [WARN] Seeds collapse to same cost (SNR={snr:.2f} < 0.3) '
              f'-- val set has low instance diversity')

    print('\n  Mode comparison on seed 1000 (round_robin vs uav_only vs adr_only):')
    for strat in ('round_robin', 'uav_only', 'adr_only'):
        r = {k: _arr(v) for k, v in
             _eval_seed(1000, args.n, cfg)[strat].items()}
        total = r['total'].mean()
        undeliv_frac = r['undeliv'].mean() / (total + 1e-9)
        print(f'    {strat:<12}: cost/req={total:.4f}  '
              f'undeliv={r["undeliv"].mean():.4f} ({undeliv_frac:.0%} of total)  '
              f'TW_UAV={r["tw_uav"].mean():.4f}  TW_ADR={r["tw_adr"].mean():.4f}')

    heavy_rr = _arr(_eval_seed(1000, args.n, cfg)['round_robin']['heavy_frac']).mean()
    print(f'\n    Avg heavy-request fraction: {heavy_rr:.1%}  '
          f'(demand > {Q_UAV}; uav_only leaves these unserved -> high undeliv cost)')
    if heavy_rr < 0.30:
        print(f'    [WARN] Only {heavy_rr:.0%} of requests are heavy — '
              f'capacity signal may be weak.  Expected ~50% with demand~U[1,10].')
    else:
        print(f'    [PASS] {heavy_rr:.0%} heavy requests — '
              f'capacity-based mode differentiation is active.')

    print('\n>>> Tight windows (tw_slack_mean=10, clip=[5,20], batched strategy)')
    tight_totals = {}

    for seed in seeds:
        print(f'\n  Seed {seed} (tight, batched):')
        res_tight = _eval_seed(seed, args.n, cfg,
                               tw_slack_mean=10.0, tw_slack_std=2.0,
                               tw_slack_clip=(5.0, 20.0),
                               strategies=('batched',))
        _summarise('batched', res_tight['batched'])
        tight_totals[seed] = _arr(res_tight['batched']['total'])

    print('\n  Normal (batched) vs tight (batched) TW penalty lift (seed 1000):')
    norm_res  = _eval_seed(1000, args.n, cfg, strategies=('batched',))
    tight_res = _eval_seed(1000, args.n, cfg,
                           tw_slack_mean=10.0, tw_slack_std=2.0,
                           tw_slack_clip=(5.0, 20.0),
                           strategies=('batched',))
    tw_norm  = _arr(norm_res['batched']['tw_uav']).mean() + \
               _arr(norm_res['batched']['tw_adr']).mean()
    tw_tight = _arr(tight_res['batched']['tw_uav']).mean() + \
               _arr(tight_res['batched']['tw_adr']).mean()
    lift = (tw_tight - tw_norm) / (tw_norm + 1e-9)
    print(f'    Normal tw_penalty : {tw_norm:.4f}/req')
    print(f'    Tight  tw_penalty : {tw_tight:.4f}/req')
    print(f'    Lift              : {lift:+.1%}')
    if lift >= 0.20:
        print(f'    [PASS] Tight windows produce {lift:.0%} more TW penalty — '
              f'val_hard_tw_penalty will differentiate HetGAT vs simplegat.')
    else:
        print(f'    [WARN] Tight windows barely lift TW penalty ({lift:.0%}) — '
              f'the hard val set may not expose encoder differences.')

    print('\n' + '=' * 65)
    print('Summary')
    print('=' * 65)
    print('  1. Cross-seed SNR tells you whether the 5 val seeds add real variance.')
    print('  2. uav_only undeliv > round_robin confirms heavy requests force ADR.')
    print('  3. Tight-window TW lift confirms val_hard_tw_penalty is a live signal.')
    print('  If all three pass, the new data generation changes are working as intended.')
    print('=' * 65)

if __name__ == '__main__':
    main()
