from __future__ import annotations
import argparse
import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def _read_csv(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

def _f(row: Dict, key: str, default=float('nan')) -> float:
    try:
        v = float(row.get(key, default))
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default

_SOLVER_DISPLAY = {
    'rl':     'RL (HetGAT)',
    'greedy': 'Greedy',
    'alns':   'ALNS',
    'gurobi': 'Gurobi',
    'ortools':'OR-Tools',
}

def _aggregate(rows: List[Dict]) -> Dict[Tuple[str, float], Dict]:
    buckets: Dict[Tuple, List[Dict]] = {}
    for r in rows:
        solver = str(r.get('baseline', r.get('solver', ''))).strip()
        delta  = _f(r, 'delta_minutes', _f(r, 'delta'))
        if not solver or math.isnan(delta):
            continue
        buckets.setdefault((solver, delta), []).append(r)

    out = {}
    for (solver, delta), grp in buckets.items():
        n_req_vals   = [_f(r, 'total_revealed', 0) for r in grp]
        cost_vals    = [_f(r, 'total_cost',     0) for r in grp]
        late_vals    = [_f(r, 'soft_time_window_violation_rate', 0) for r in grp]
        dec_ms_vals  = [_f(r, 'solver_time_s_mean', 0) * 1000 for r in grp]
        shift_s_vals = [_f(r, 'wall_time_s', 0) for r in grp]

        n_req_mean = float(np.nanmean(n_req_vals)) or 1.0
        out[(solver, delta)] = {
            'cost_per_req': float(np.nanmean(cost_vals)) / n_req_mean,
            'ontime_pct':   1.0 - float(np.nanmean(late_vals)),
            'decision_ms':  float(np.nanmean(dec_ms_vals)),
            'shift_s':      float(np.nanmean(shift_s_vals)),
            'n':            len(grp),
        }
    return out

def plot_comparison_table(
    agg: Dict[Tuple, Dict],
    deltas: List[float],
    out_dir: Path,
) -> Optional[Path]:
    solvers = sorted({s for s, _ in agg}, key=lambda s: ('rl', 'greedy', 'alns').index(s)
                     if s in ('rl', 'greedy', 'alns') else 99)

    if not solvers:
        print('  [SKIP] comparison table: no aggregated data')
        return None

    col_groups = [f'Δ={int(d)} min' for d in deltas]
    metrics    = ['Cost/req', 'On-time %', 'Decision (ms)', 'Shift (s)']
    col_labels = []
    for g in col_groups:
        for m in metrics:
            col_labels.append(f'{g}\n{m}')

    row_labels = [_SOLVER_DISPLAY.get(s, s) for s in solvers]
    table_data = []

    def _fmt(key, val):
        if math.isnan(val):
            return '—'
        if key == 'ontime_pct':    return f'{val:.1%}'
        if key == 'cost_per_req':  return f'{val:.3f}'
        if key == 'decision_ms':   return f'{val:.0f}'
        if key == 'shift_s':       return f'{val:.1f}'
        return f'{val:.3f}'

    for solver in solvers:
        row = []
        for delta in deltas:
            data = agg.get((solver, delta), {})
            for key in ('cost_per_req', 'ontime_pct', 'decision_ms', 'shift_s'):
                row.append(_fmt(key, data.get(key, float('nan'))))
        table_data.append(row)

    n_rows = len(table_data)
    n_cols = len(col_labels)
    cell_colors = [['white'] * n_cols for _ in range(n_rows)]

    metric_keys = ['cost_per_req', 'ontime_pct', 'decision_ms', 'shift_s']
    for di, delta in enumerate(deltas):
        for mi, mkey in enumerate(metric_keys):
            col_idx = di * len(metrics) + mi
            vals = [agg.get((s, delta), {}).get(mkey, float('nan')) for s in solvers]
            if all(math.isnan(v) for v in vals):
                continue
            best_row = (np.nanargmax(vals) if mkey == 'ontime_pct'
                        else np.nanargmin(vals))
            cell_colors[best_row][col_idx] = '#d4edda'

    row_colors_left = [['#eef4fb'] * 1 for _ in range(n_rows)]
    all_colors      = [[rc[0]] + cc for rc, cc in zip(row_colors_left, cell_colors)]

    fig_w = max(14, len(col_labels) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 2.2 + 0.45 * n_rows))
    ax.axis('off')

    tbl = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=all_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.55)
    ax.set_title(
        'On-demand dispatch: (RL vs ALNS vs Greedy) × Δ ∈ {5, 10, 15} min\n'
        'Green = best per column.  Cost/req normalised by revealed requests.',
        fontsize=10, pad=16,
    )

    p = out_dir / 'delta_comparison_table.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return p

def plot_training_amortisation(
    training_hours_rl: float,
    rl_ms: float,
    alns_ms: float,
    greedy_ms: float,
    out_dir: Path,
    max_decisions: int = 2_000_000,
) -> Path:
    train_s   = training_hours_rl * 3600.0
    rl_per    = rl_ms / 1000.0
    alns_per  = alns_ms / 1000.0
    greedy_per= greedy_ms / 1000.0

    crossover_alns   = train_s / max(alns_per - rl_per, 1e-6)
    crossover_greedy = train_s / max(greedy_per - rl_per, 1e-6) if greedy_per > rl_per else None

    N = np.linspace(0, max_decisions, 2000)

    cost_rl     = train_s + N * rl_per
    cost_alns   = N * alns_per
    cost_greedy = N * greedy_per

    H = 3600.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N / 1e3, cost_rl     / H, label=f'RL (HetGAT)  {rl_ms:.0f} ms/dec',
            color='#2ca02c', linewidth=2)
    ax.plot(N / 1e3, cost_alns   / H, label=f'ALNS  {alns_ms:.0f} ms/dec',
            color='darkorange', linewidth=2, linestyle='--')
    ax.plot(N / 1e3, cost_greedy / H, label=f'Greedy  {greedy_ms:.0f} ms/dec',
            color='steelblue', linewidth=2, linestyle=':')

    if 0 < crossover_alns < max_decisions:
        xc_h = (train_s + crossover_alns * rl_per) / H
        ax.axvline(crossover_alns / 1e3, color='grey', linewidth=1.0, linestyle='--')
        ax.annotate(
            f'RL cheaper\nthan ALNS\nafter {crossover_alns/1e3:.0f}k decisions',
            xy=(crossover_alns / 1e3, xc_h),
            xytext=(crossover_alns / 1e3 + max_decisions * 0.04 / 1e3, xc_h * 1.1),
            fontsize=8, color='grey',
            arrowprops=dict(arrowstyle='->', color='grey'),
        )

    ax.set_xlabel('Cumulative dispatch decisions (thousands)', fontsize=10)
    ax.set_ylabel('Total compute time (hours)', fontsize=10)
    ax.set_title(
        'Training amortisation: when does RL pay off?\n'
        f'Training cost = {training_hours_rl:.0f} h  '
        f'(RL: {rl_ms:.0f} ms/dec, ALNS: {alns_ms:.0f} ms/dec, Greedy: {greedy_ms:.0f} ms/dec)',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f h'))

    p = out_dir / 'training_amortisation.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

_RUNG_FLEET = {
    'A': dict(n_req=5,  n_uav=2,  n_adr=2, n_depots_uav=1, n_depots_adr=1),
    'B': dict(n_req=10, n_uav=4,  n_adr=3, n_depots_uav=1, n_depots_adr=1),
    'C': dict(n_req=25, n_uav=5,  n_adr=4, n_depots_uav=2, n_depots_adr=2),
    'D': dict(n_req=60, n_uav=10, n_adr=8, n_depots_uav=3, n_depots_adr=3),
}

def _run_benchmark(rl_model: str, rung: str, delta: float, n_instances: int,
                   out_dir: Path, arch: str = 'hetgat',
                   seed: int = 9999, include_gurobi: bool = False,
                   ortools_solver: str = 'GLPK',
                   demand_range=(1.0, 6.0), wind_speed_range=(0.0, 12.0),
                   tw_slack_mean=30.0, tw_slack_std=5.0,
                   tw_slack_clip=(15.0, 60.0),
                   scenario_label: str = 'baseline') -> str:
    fleet = _RUNG_FLEET[rung]
    bm_out = out_dir / f'bm_{rung}_delta{int(delta)}'
    bm_out.mkdir(parents=True, exist_ok=True)

    if rung == 'D' or not include_gurobi:
        cpu_bl = 'fifo,greedy,alns,offline_alns'
    else:
        cpu_bl = 'fifo,greedy,alns,offline_alns,gurobi'

    baselines = ('rl,' + cpu_bl) if rl_model else cpu_bl

    cmd = [
        sys.executable, 'benchmark_rolling_baselines_with_plots.py',
        '--delta-minutes',   str(delta),
        '--num-instances',   str(n_instances),
        '--n-req',           str(fleet['n_req']),
        '--n-uav',           str(fleet['n_uav']),
        '--n-adr',           str(fleet['n_adr']),
        '--n-depots-uav',    str(fleet['n_depots_uav']),
        '--n-depots-adr',    str(fleet['n_depots_adr']),
        '--baselines',       baselines,
        '--seed',            str(seed),
        '--output-dir',      str(bm_out),
        '--ortools-solver',       ortools_solver,
        '--demand-low',           str(demand_range[0]),
        '--demand-high',          str(demand_range[1]),
        '--wind-speed-low',       str(wind_speed_range[0]),
        '--wind-speed-high',      str(wind_speed_range[1]),
        '--tw-slack-mean',        str(tw_slack_mean),
        '--tw-slack-std',         str(tw_slack_std),
        '--tw-slack-clip-low',    str(tw_slack_clip[0]),
        '--tw-slack-clip-high',   str(tw_slack_clip[1]),
        '--scenario-label',       scenario_label,
        '--workers',              '1',
        '--make-plots',
    ]
    if rl_model:
        cmd += ['--rl-model', rl_model, '--arch', arch]

    print(f'  Running benchmark Δ={delta} min  rung={rung}  arch={arch}  scenario={scenario_label} → {bm_out}')
    ret = subprocess.run(cmd, cwd='.')
    if ret.returncode != 0:
        print(f'  WARNING: benchmark exited with code {ret.returncode}')

    return str(bm_out / 'per_instance_results.csv')

def main():
    parser = argparse.ArgumentParser(description='Claim 3 evaluation figures')
    parser.add_argument('--rl-model',  default='',
                        help='Path to best_actor.pt (triggers benchmark runs if --from-csv not given)')
    parser.add_argument('--arch',      default='hetgat', choices=['hetgat', 'simplegat'],
                        help='Architecture of the RL checkpoint (hetgat or simplegat)')
    parser.add_argument('--rung',      default='C', choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--n-instances', type=int, default=100,
                        help='Instances per (solver, delta) combination')
    parser.add_argument('--deltas',    type=float, nargs='+', default=[5.0, 10.0, 15.0])
    parser.add_argument('--seed',      type=int, default=9999,
                        help='Random seed (must match seed used in submit_baseline.sh)')
    parser.add_argument('--gurobi',    action='store_true',
                        help='Include Gurobi in the sweep for Rungs A/B/C (slow; skip for D)')
    parser.add_argument('--ortools-solver', default='GLPK',
                        help='OR-Tools backend: GLPK (bundled, safe on CC) or SCIP (ABI issues on Narval)')
    parser.add_argument('--from-csv',  nargs='+', default=[],
                        help='Existing benchmark CSV files (skips re-running)')
    parser.add_argument('--out-dir',   default='eval_results/plots/claim3')
    parser.add_argument('--training-hours', type=float, default=18.0,
                        help='Estimated RL training time in hours')
    parser.add_argument('--rl-ms',    type=float, default=None,
                        help='Override RL per-decision time in ms (default: from benchmark)')
    parser.add_argument('--alns-ms',  type=float, default=None,
                        help='Override ALNS per-decision time in ms')
    parser.add_argument('--greedy-ms',type=float, default=None,
                        help='Override greedy per-decision time in ms')
    parser.add_argument('--amortisation-only', action='store_true',
                        help='Skip table; only produce amortisation chart')
    parser.add_argument('--demand-low',  type=float, default=1.0)
    parser.add_argument('--demand-high', type=float, default=6.0)
    parser.add_argument('--wind-speed-low',  type=float, default=0.0)
    parser.add_argument('--wind-speed-high', type=float, default=12.0)
    parser.add_argument('--tw-slack-mean', type=float, default=30.0)
    parser.add_argument('--tw-slack-std',  type=float, default=5.0)
    parser.add_argument('--tw-slack-clip-low',  type=float, default=15.0)
    parser.add_argument('--tw-slack-clip-high', type=float, default=60.0)
    parser.add_argument('--scenario-label', default='baseline',
                        help='Label for this scenario (used in output dirs and table headers)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []

    if args.from_csv:
        for csv_path in args.from_csv:
            rows = _read_csv(csv_path)
            print(f'Read {len(rows)} rows from {csv_path}')
            all_rows.extend(rows)
    elif not args.amortisation_only:
        if not args.rl_model:
            parser.error('Provide --rl-model or --from-csv (or --amortisation-only)')
        bm_dir = Path('eval_results')
        bm_dir.mkdir(exist_ok=True)
        for delta in args.deltas:
            csv_path = _run_benchmark(
                args.rl_model, args.rung, delta, args.n_instances, bm_dir,
                arch=args.arch, seed=args.seed,
                include_gurobi=getattr(args, 'gurobi', False),
                ortools_solver=args.ortools_solver,
                demand_range=(args.demand_low, args.demand_high),
                wind_speed_range=(args.wind_speed_low, args.wind_speed_high),
                tw_slack_mean=args.tw_slack_mean,
                tw_slack_std=args.tw_slack_std,
                tw_slack_clip=(args.tw_slack_clip_low, args.tw_slack_clip_high),
                scenario_label=args.scenario_label,
            )
            all_rows.extend(_read_csv(csv_path))

    if all_rows and not args.amortisation_only:
        agg = _aggregate(all_rows)
        print(f'\nAggregated {len(agg)} (solver, delta) combinations:')
        for (solver, delta), stats in sorted(agg.items()):
            print(f'  {solver:<10} Δ={delta:4.0f}  '
                  f'cost/req={stats["cost_per_req"]:.3f}  '
                  f'ontime={stats["ontime_pct"]:.1%}  '
                  f'dec={stats["decision_ms"]:.0f}ms  '
                  f'shift={stats["shift_s"]:.1f}s  '
                  f'n={stats["n"]}')

        p = plot_comparison_table(agg, sorted(args.deltas), out_dir)
        if p: print(f'Saved: {p}')

        if args.rl_ms is None:
            rl_rows  = [(s, d) for (s, d) in agg if s == 'rl']
            args.rl_ms = float(np.mean([agg[k]['decision_ms'] for k in rl_rows])) if rl_rows else 2.0
        if args.alns_ms is None:
            alns_rows = [(s, d) for (s, d) in agg if s == 'alns']
            args.alns_ms = float(np.mean([agg[k]['decision_ms'] for k in alns_rows])) if alns_rows else 4000.0
        if args.greedy_ms is None:
            g_rows = [(s, d) for (s, d) in agg if s == 'greedy']
            args.greedy_ms = float(np.mean([agg[k]['decision_ms'] for k in g_rows])) if g_rows else 10.0

    rl_ms     = args.rl_ms     or 2.0
    alns_ms   = args.alns_ms   or 4000.0
    greedy_ms = args.greedy_ms or 10.0

    p = plot_training_amortisation(
        training_hours_rl=args.training_hours,
        rl_ms=rl_ms, alns_ms=alns_ms, greedy_ms=greedy_ms,
        out_dir=out_dir,
    )
    print(f'Saved: {p}')
    print('\nDone.')

if __name__ == '__main__':
    main()
