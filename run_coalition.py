from __future__ import annotations
import argparse
import csv
import os
import time
from typing import Dict, List, Any, Optional

import numpy as np
from scipy.stats import pearsonr

from coalition import run_full_sweep, SWEEP_AXES
from allocation import conviction_score
from greedy_insertion import GreedyInsertion

def _scenario_label(config: Dict[str, Any]) -> str:
    ms  = config['market_split']
    uav = int(ms['uav'] * 100)
    adr = 100 - uav
    eps = int(config['eps_shared'] * 100)
    dep = 'shared' if config['depot_shared'] else 'sep'
    aa  = f"a{config['alpha_asymmetry']:.2f}"
    cu  = config.get('charging_cost_uav', 0.0)
    ca  = config.get('charging_cost_adr', 0.0)
    return f'U{uav}A{adr}_e{eps}_{dep}_{aa}_cu{cu:.0f}ca{ca:.0f}'

def _winner(phi_uav: float, phi_adr: float, phi_coal: float) -> str:
    best = min(phi_uav, phi_adr, phi_coal)
    if best == phi_coal:
        return 'coalition'
    if best == phi_uav:
        return 'uav_only'
    return 'adr_only'

def _factor_sensitivity(results: List[Dict[str, Any]]) -> None:
    gains = np.array([r['coalition_gain_mean'] for r in results])

    factors = {
        'uav_market_frac':   [r['config']['market_split']['uav'] for r in results],
        'eps_shared':        [r['config']['eps_shared'] for r in results],
        'depot_shared':      [float(r['config']['depot_shared']) for r in results],
        'alpha_asymmetry':   [r['config']['alpha_asymmetry'] for r in results],
        'charging_cost_uav': [r['config'].get('charging_cost_uav', 0.0) for r in results],
        'charging_cost_adr': [r['config'].get('charging_cost_adr', 0.0) for r in results],
    }

    print('\n=== Factor sensitivity (Pearson r with coalition_gain_mean) ===')
    print('  (Positive r = higher value => more coalition gain)')
    corrs = []
    for fname, fvals in factors.items():
        fv = np.array(fvals, dtype=float)
        if fv.std() < 1e-9:
            continue
        r_val, p_val = pearsonr(fv, gains)
        corrs.append((fname, r_val, p_val))

    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for fname, r_val, p_val in corrs:
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else '(ns)'))
        print(f'  {fname:<22}: r={r_val:+.3f}  p={p_val:.4f}  {sig}')

def _per_firm_stability_table(results: List[Dict[str, Any]]) -> None:
    methods = ('shapley', 'epm', 'pam')
    print('\n=== Per-firm individual rationality (averaged over all scenarios) ===')
    print(f'  {"Method":<10}  {"UAV benefits":<20}  {"ADR benefits":<20}  '
          f'{"Both benefit":<20}  UAV p5-gain  ADR p5-gain')
    print('  ' + '-' * 100)
    for m in methods:
        uav_ir  = np.mean([r.get(f'uav_ir_rate_{m}',  0) for r in results])
        adr_ir  = np.mean([r.get(f'adr_ir_rate_{m}',  0) for r in results])
        core    = np.mean([r.get(f'core_rate_{m}',    0) for r in results])
        uav_p5  = np.mean([r.get(f'uav_gain_p5_{m}',  0) for r in results])
        adr_p5  = np.mean([r.get(f'adr_gain_p5_{m}',  0) for r in results])
        print(f'  {m.upper():<10}  {uav_ir:.1%} of episodes    {adr_ir:.1%} of episodes    '
              f'{core:.1%} of episodes  {uav_p5:+.2f}       {adr_p5:+.2f}')

    print('\n  Core rate (Shapley) by UAV market share:')
    for frac_str, frac in [('UAV30%', 0.30), ('UAV50%', 0.50), ('UAV70%', 0.70)]:
        subset = [r for r in results if abs(r['config']['market_split']['uav'] - frac) < 0.01]
        if subset:
            core_vals = [r.get('core_rate_shapley', 0) for r in subset]
            uav_gain  = [r.get('uav_gain_mean_shapley', 0) for r in subset]
            adr_gain  = [r.get('adr_gain_mean_shapley', 0) for r in subset]
            print(f'    {frac_str}: core={np.mean(core_vals):.1%}  '
                  f'UAV gain={np.mean(uav_gain):.2f}  ADR gain={np.mean(adr_gain):.2f}')

    print('\n  Core rate (Shapley) by depot configuration:')
    for shared in (False, True):
        subset = [r for r in results if r['config']['depot_shared'] == shared]
        if subset:
            core_vals = [r.get('core_rate_shapley', 0) for r in subset]
            gain_vals = [r.get('coalition_gain_mean', 0) for r in subset]
            label = 'shared depots  ' if shared else 'separate depots'
            print(f'    {label}: core={np.mean(core_vals):.1%}  '
                  f'mean gain={np.mean(gain_vals):.2f}')

def _decision_table(results: List[Dict[str, Any]], core_threshold: float = 0.70) -> None:
    eligible = [
        r for r in results
        if r.get('core_rate_shapley', 0) >= core_threshold
    ]
    eligible.sort(key=lambda r: r.get('conviction', 0), reverse=True)

    print(f'\n=== Scenarios where both firms benefit ≥{core_threshold:.0%} of episodes ===')
    print(f'  ({len(eligible)} of {len(results)} scenarios qualify)')
    if not eligible:
        print(f'  None found at threshold {core_threshold:.0%}. '
              f'Best core rate: {max(r.get("core_rate_shapley",0) for r in results):.1%}')
        return

    header = f'  {"Scenario":<42} {"CoreRate":>9} {"MeanGain":>10} {"UAV p5":>8} {"ADR p5":>8} {"Conviction":>11}'
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for r in eligible[:20]:
        cfg = r['config']
        lbl = _scenario_label(cfg)
        cr  = r.get('core_rate_shapley', 0)
        mg  = r.get('coalition_gain_mean', 0)
        up5 = r.get('uav_gain_p5_shapley', 0)
        ap5 = r.get('adr_gain_p5_shapley', 0)
        cv  = r.get('conviction', 0)
        print(f'  {lbl:<42} {cr:>9.1%} {mg:>10.2f} {up5:>+8.2f} {ap5:>+8.2f} {cv:>11.3f}')

def _compare_solvers(
    greedy_results: List[Dict[str, Any]],
    rl_results: List[Dict[str, Any]],
    greedy_time_s: float,
    rl_time_s: float,
) -> None:
    n = len(greedy_results)
    assert len(rl_results) == n, 'Solver results must cover identical scenario set'

    metrics = [
        ('coalition_gain_mean',   'Mean total gain',    '{:+.3f}'),
        ('core_rate_shapley',     'Core rate (Shapley)', '{:.1%}'),
        ('uav_ir_rate_shapley',   'UAV IR rate',         '{:.1%}'),
        ('adr_ir_rate_shapley',   'ADR IR rate',         '{:.1%}'),
        ('conviction',            'Mean conviction',     '{:.3f}'),
        ('coalition_gain_p5',     'Gain p5 (worst-case)','{:+.3f}'),
    ]

    g_vals = {k: np.mean([r.get(k, 0) for r in greedy_results]) for k, *_ in metrics}
    r_vals = {k: np.mean([r.get(k, 0) for r in rl_results])    for k, *_ in metrics}

    print('\n=== RL vs GreedyInsertion: coalition performance ===')
    print(f'  {"Metric":<30}  {"Greedy":>10}  {"RL":>10}  {"Δ (RL-Greedy)":>15}')
    print('  ' + '-' * 70)
    for key, label, fmt in metrics:
        gv = g_vals[key]
        rv = r_vals[key]
        delta = rv - gv
        fmt_g = fmt.format(gv)
        fmt_r = fmt.format(rv)
        fmt_d = f'{delta:+.3f}' if isinstance(delta, float) else str(delta)
        arrow  = '+' if delta > 0 else ('-' if delta < 0 else '=')
        print(f'  {label:<30}  {fmt_g:>10}  {fmt_r:>10}  {fmt_d:>12} {arrow}')

    rl_better_gain = sum(
        1 for g, r in zip(greedy_results, rl_results)
        if r.get('coalition_gain_mean', 0) > g.get('coalition_gain_mean', 0)
    )
    rl_better_core = sum(
        1 for g, r in zip(greedy_results, rl_results)
        if r.get('core_rate_shapley', 0) > g.get('core_rate_shapley', 0)
    )
    print(f'\n  RL improves coalition gain in {rl_better_gain}/{n} scenarios ({rl_better_gain/n:.0%})')
    print(f'  RL improves core stability  in {rl_better_core}/{n} scenarios ({rl_better_core/n:.0%})')

    n_ep = 1
    print(f'\n  Solve time (full sweep): Greedy={greedy_time_s:.0f}s  RL={rl_time_s:.0f}s')
    if rl_time_s > 0 and greedy_time_s > 0:
        ratio = rl_time_s / greedy_time_s
        print(f'  RL is {ratio:.1f}x {"slower" if ratio > 1 else "faster"} than Greedy  '
              f'(amortised over full deployment, inference cost is negligible vs training)')

def analyse(results: List[Dict[str, Any]]) -> None:
    winners: Dict[str, int] = {'uav_only': 0, 'adr_only': 0, 'coalition': 0}
    gain_by_uav_frac: Dict[str, List[float]] = {}
    gain_by_eps: Dict[float, List[float]]   = {}
    gain_by_depot: Dict[bool, List[float]]  = {}
    gain_by_infra: Dict[str, List[float]]   = {}

    for r in results:
        cfg  = r['config']
        w    = _winner(r['phi_uav_mean'], r['phi_adr_mean'], r['phi_coalition_mean'])
        gain = r['coalition_gain_mean']
        winners[w] += 1

        uav_frac = cfg['market_split']['uav']
        gain_by_uav_frac.setdefault(f"UAV{int(uav_frac*100)}%", []).append(gain)
        gain_by_eps.setdefault(cfg['eps_shared'], []).append(gain)
        gain_by_depot.setdefault(cfg['depot_shared'], []).append(gain)
        cu = cfg.get('charging_cost_uav', 0.0)
        ca = cfg.get('charging_cost_adr', 0.0)
        gain_by_infra.setdefault(f'uav={cu:.0f} adr={ca:.0f}', []).append(gain)

    n = len(results)
    print('\n=== Coalition Analysis ===')
    print(f'Total scenarios: {n}')
    print(f'\nWinner distribution:')
    for k, v in winners.items():
        print(f'  {k:<15}: {v:3d}/{n}  ({100*v/n:.0f}%)')

    print(f'\nMean coalition gain by UAV market share:')
    for k in sorted(gain_by_uav_frac):
        vals = gain_by_uav_frac[k]
        print(f'  {k}: gain={np.mean(vals):.4f} ± {np.std(vals):.4f}')

    print(f'\nMean coalition gain by cross-mode tolerance (eps_shared):')
    for k in sorted(gain_by_eps):
        vals = gain_by_eps[k]
        print(f'  eps={k:.2f}: gain={np.mean(vals):.4f} ± {np.std(vals):.4f}')

    print(f'\nMean coalition gain by depot configuration:')
    for k, vals in sorted(gain_by_depot.items()):
        label = 'shared depots  ' if k else 'separate depots'
        print(f'  {label}: gain={np.mean(vals):.4f} ± {np.std(vals):.4f}')

    print(f'\nMean coalition gain by charging infrastructure cost:')
    for k in sorted(gain_by_infra):
        vals = gain_by_infra[k]
        print(f'  {k}: gain={np.mean(vals):.4f} ± {np.std(vals):.4f}')

    n_core = sum(1 for r in results if r['core_exists'])
    core_freq_mean = np.mean([r.get('core_freq', float(r['core_exists'])) for r in results])
    print(f'\nCore-stable scenarios (coalition profitable >=50% of episodes): '
          f'{n_core}/{n} ({100*n_core/n:.0f}%)')
    print(f'Mean core frequency: {core_freq_mean:.3f}')

    if 'conviction' in results[0]:
        conv_vals = [r['conviction'] for r in results]
        print(f'\nConviction score (higher = stronger joint business case):')
        print(f'  Mean={np.mean(conv_vals):.3f}  Max={np.max(conv_vals):.3f}  '
              f'Scenarios > 0.5: {sum(c > 0.5 for c in conv_vals)}/{n}')

    _per_firm_stability_table(results)

    _factor_sensitivity(results)

    _decision_table(results, core_threshold=0.70)
    _decision_table(results, core_threshold=0.50)

def _flatten_result(r: Dict[str, Any]) -> Dict[str, Any]:
    cfg = r['config']
    row: Dict[str, Any] = {
        'scenario':             _scenario_label(cfg),
        'uav_market_frac':      cfg['market_split']['uav'],
        'adr_market_frac':      cfg['market_split']['adr'],
        'eps_shared':           cfg['eps_shared'],
        'depot_shared':         int(cfg['depot_shared']),
        'alpha_asymmetry':      cfg['alpha_asymmetry'],
        'charging_cost_uav':    cfg.get('charging_cost_uav', 0.0),
        'charging_cost_adr':    cfg.get('charging_cost_adr', 0.0),
        'delta_minutes':        cfg.get('delta_minutes', 10.0),
        'phi_uav_mean':         r['phi_uav_mean'],
        'phi_uav_std':          r['phi_uav_std'],
        'phi_adr_mean':         r['phi_adr_mean'],
        'phi_adr_std':          r['phi_adr_std'],
        'phi_coalition_mean':   r['phi_coalition_mean'],
        'phi_coalition_std':    r['phi_coalition_std'],
        'coalition_gain_mean':  r['coalition_gain_mean'],
        'coalition_gain_std':   r.get('coalition_gain_std', ''),
        'coalition_gain_p5':          r.get('coalition_gain_p5', ''),
        'coalition_gain_positive_rate': r.get('coalition_gain_positive_rate', ''),
        'core_exists':          int(r['core_exists']),
        'core_freq':            r.get('core_freq', float(r['core_exists'])),
        'conviction':           r.get('conviction', ''),
        'mean_undelivered':     r.get('mean_undelivered', 0.0),
        'winner':               _winner(r['phi_uav_mean'], r['phi_adr_mean'], r['phi_coalition_mean']),
    }
    for m in ('shapley', 'epm', 'pam'):
        row[f'uav_ir_rate_{m}']   = r.get(f'uav_ir_rate_{m}', '')
        row[f'adr_ir_rate_{m}']   = r.get(f'adr_ir_rate_{m}', '')
        row[f'core_rate_{m}']     = r.get(f'core_rate_{m}', '')
        row[f'uav_gain_mean_{m}'] = r.get(f'uav_gain_mean_{m}', '')
        row[f'adr_gain_mean_{m}'] = r.get(f'adr_gain_mean_{m}', '')
        row[f'uav_gain_p5_{m}']   = r.get(f'uav_gain_p5_{m}', '')
        row[f'adr_gain_p5_{m}']   = r.get(f'adr_gain_p5_{m}', '')

    row['uav_benefit_mean'] = r.get('uav_benefit_mean', '')
    row['adr_benefit_mean'] = r.get('adr_benefit_mean', '')

    row['n_req']                = r.get('n_req', '')
    row['uav_late_pct_mean']    = r.get('uav_late_pct_mean', '')
    row['adr_late_pct_mean']    = r.get('adr_late_pct_mean', '')
    row['coal_late_pct_mean']   = r.get('coal_late_pct_mean', '')
    row['uav_served_frac_mean'] = r.get('uav_served_frac_mean', '')
    row['adr_served_frac_mean'] = r.get('adr_served_frac_mean', '')
    row['coal_served_frac_mean']= r.get('coal_served_frac_mean', '')
    row['uav_cost_per_req']     = r.get('uav_cost_per_req', '')
    row['adr_cost_per_req']     = r.get('adr_cost_per_req', '')
    row['coal_cost_per_req']    = r.get('coal_cost_per_req', '')
    return row

def _save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Saved: {path}')

def main() -> None:
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    _RUNG_FLEET = {
        'A': dict(n_req=5,  n_uav=2,  n_adr=2),
        'B': dict(n_req=10, n_uav=4,  n_adr=3),
        'C': dict(n_req=25, n_uav=5,  n_adr=4),
        'D': dict(n_req=60, n_uav=10, n_adr=8),
    }

    parser = argparse.ArgumentParser(
        description='CE-PDPTW coalition cost sweep',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--episodes', type=int, default=50,
                        help='Episodes per scenario (use 200 for paper results)')
    parser.add_argument('--delta', type=float, default=10.0,
                        help='Rolling-horizon delta in minutes')
    parser.add_argument('--seed', type=int, default=9999, help='Base random seed')
    parser.add_argument('--out-dir', default='eval_results', help='Output directory')
    parser.add_argument('--rung', default='C', choices=list(_RUNG_FLEET),
                        help='Curriculum rung — sets fleet size (n_req/n_uav/n_adr). '
                             'Run once per rung to cover all problem scales.')
    parser.add_argument('--solver', default='greedy',
                        choices=['greedy', 'alns', 'gurobi', 'ortools'],
                        help='Solver for the primary sweep (greedy | alns | gurobi | ortools)')
    parser.add_argument('--rl-model', default='',
                        help='Path to best_actor.pt for RL vs Greedy comparison (always runs RL second)')
    parser.add_argument('--rl-episodes', type=int, default=None,
                        help='Episodes for RL comparison (default: same as --episodes)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel worker processes for the sweep '
                             '(default 1; set to cpu_count on Narval, e.g. 16). '
                             'RL solver must be CPU-only when workers > 1.')
    args = parser.parse_args()

    fleet = _RUNG_FLEET[args.rung]
    n_req, n_uav, n_adr = fleet['n_req'], fleet['n_uav'], fleet['n_adr']

    os.makedirs(args.out_dir, exist_ok=True)

    n_scenarios = (
        len(SWEEP_AXES['market_split'])
        * len(SWEEP_AXES['eps_shared'])
        * len(SWEEP_AXES['depot_shared'])
        * len(SWEEP_AXES['alpha_asymmetry'])
        * len(SWEEP_AXES['charging_infra'])
    )
    total_runs = n_scenarios * args.episodes
    print(f'Rung {args.rung}: n_req={n_req}  n_uav={n_uav}  n_adr={n_adr}')
    print(f'Sweep: {n_scenarios} scenarios × {args.episodes} episodes = {total_runs} dispatcher runs')

    solver_name = args.solver
    if solver_name == 'greedy':
        primary_solver = GreedyInsertion()
    elif solver_name == 'alns':
        from ce_cpdptw_alns import ALNSBaseline
        primary_solver = ALNSBaseline()
    elif solver_name == 'gurobi':
        from ce_cpdptw_rolling_baselines import GurobiRollingHorizonSolver, ExactRollingConfig
        primary_solver = GurobiRollingHorizonSolver(ExactRollingConfig())
    elif solver_name == 'ortools':
        from ce_cpdptw_rolling_baselines import ORToolsRollingHorizonSolver, ExactRollingConfig
        primary_solver = ORToolsRollingHorizonSolver(ExactRollingConfig())
    else:
        raise ValueError(f'Unknown solver: {solver_name}')

    print(f'Solver: {solver_name}, delta={args.delta} min')
    greedy_solver = primary_solver

    t0 = time.time()
    greedy_results = run_full_sweep(
        solver=greedy_solver,
        n_episodes=args.episodes,
        delta_minutes=args.delta,
        seed_base=args.seed,
        n_workers=args.workers,
        n_req=n_req,
        n_uav=n_uav,
        n_adr=n_adr,
    )
    greedy_time = time.time() - t0
    print(f'{solver_name} sweep done in {greedy_time:.0f}s ({greedy_time/max(total_runs,1):.2f}s/run)')

    analyse(greedy_results)

    rows_greedy = [_flatten_result(r) for r in greedy_results]
    rows_sorted = sorted(rows_greedy, key=lambda r: float(r.get('conviction', 0) or 0), reverse=True)
    print('\n=== Top 5 scenarios by conviction score (strongest joint business case) ===')
    for i, row in enumerate(rows_sorted[:5]):
        cr  = row.get('core_rate_shapley', '')
        up5 = row.get('uav_gain_p5_shapley', '')
        ap5 = row.get('adr_gain_p5_shapley', '')
        print(
            f'{i+1}. {row["scenario"]}\n'
            f'   conviction={float(row["conviction"] or 0):.3f}  '
            f'gain={row["coalition_gain_mean"]:.3f}  '
            f'core(Shapley)={float(cr or 0):.1%}  '
            f'winner={row["winner"]}\n'
            f'   UAV p5-gain={float(up5 or 0):+.2f}  ADR p5-gain={float(ap5 or 0):+.2f}  '
            f'undelivered={row["mean_undelivered"]:.2f}'
        )

    print('\n=== Bottom 5 scenarios (coalition least beneficial) ===')
    for i, row in enumerate(rows_sorted[-5:]):
        print(
            f'  {row["scenario"]}: '
            f'gain={row["coalition_gain_mean"]:.3f}  '
            f'conviction={float(row.get("conviction", 0) or 0):.3f}  '
            f'winner={row["winner"]}'
        )

    greedy_csv = os.path.join(args.out_dir, f'coalition_sweep_{solver_name}.csv')
    _save_csv(rows_greedy, greedy_csv)

    rl_results: Optional[List[Dict[str, Any]]] = None
    rl_time = 0.0
    if args.rl_model:
        from main import build_solver as _build_rl
        rl_solver = _build_rl(args.rl_model)
        rl_ep = args.rl_episodes or args.episodes
        rl_total = n_scenarios * rl_ep
        print(f'\nRL comparison: {n_scenarios} scenarios × {rl_ep} episodes = {rl_total} runs')
        t0 = time.time()
        rl_results = run_full_sweep(
            solver=rl_solver,
            n_episodes=rl_ep,
            delta_minutes=args.delta,
            seed_base=args.seed,
            n_workers=1,
            n_req=n_req,
            n_uav=n_uav,
            n_adr=n_adr,
        )
        rl_time = time.time() - t0
        print(f'RL sweep done in {rl_time:.0f}s ({rl_time/max(rl_total,1):.2f}s/run)')

        _compare_solvers(greedy_results, rl_results, greedy_time, rl_time)

        rows_rl = [_flatten_result(r) for r in rl_results]
        _save_csv(rows_rl, os.path.join(args.out_dir, 'coalition_sweep_rl.csv'))

        merged = []
        for g, r in zip(rows_greedy, rows_rl):
            row = dict(g)
            for k in ('coalition_gain_mean', 'core_rate_shapley', 'conviction',
                      'uav_ir_rate_shapley', 'adr_ir_rate_shapley'):
                gv = float(g.get(k, 0) or 0)
                rv = float(r.get(k, 0) or 0)
                row[f'rl_{k}'] = rv
                row[f'delta_{k}'] = rv - gv
            merged.append(row)
        _save_csv(merged, os.path.join(args.out_dir, f'coalition_comparison_rl_vs_{solver_name}.csv'))

if __name__ == '__main__':
    main()
