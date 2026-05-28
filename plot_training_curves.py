from __future__ import annotations
import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

def _col(rows: List[Dict], key: str) -> List[float]:
    out = []
    for r in rows:
        try:
            v = float(r[key])
            if math.isfinite(v):
                out.append(v)
            else:
                out.append(float('nan'))
        except (KeyError, ValueError, TypeError):
            out.append(float('nan'))
    return out

def _valid(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
    if not pairs:
        return [], []
    return zip(*pairs)

def _mark_baseline_updates(ax, epochs, updated_flags):
    for ep, flag in zip(epochs, updated_flags):
        try:
            if float(flag) > 0.5:
                ax.axvline(ep, color='green', alpha=0.3, linewidth=0.8, linestyle='--')
        except (ValueError, TypeError):
            pass

def plot_training_curves(
    log_path: Path,
    out_dir: Path,
    rung: str,
) -> List[Path]:
    rows = _read_csv(log_path)
    if not rows:
        print(f'  [SKIP] No training log found: {log_path}')
        return []

    out_dir.mkdir(parents=True, exist_ok=True)

    epochs       = _col(rows, 'epoch')
    val_costs    = _col(rows, 'val_cost')
    rewards      = _col(rows, 'train_reward_mean')
    losses       = _col(rows, 'train_loss')
    grad_means   = _col(rows, 'grad_norm_mean')
    grad_maxes   = _col(rows, 'grad_norm_max')
    bl_updated   = [r.get('baseline_updated', 0) for r in rows]

    created = []

    specs = [
        ('training_cost.png',     'Validation cost',       val_costs,  'Cost'),
        ('training_reward.png',   'Mean training reward',  rewards,    'Reward'),
        ('training_loss.png',     'Actor loss',            losses,     'Loss'),
        ('training_gradnorm.png', 'Gradient norm (mean)',  grad_means, 'Grad norm'),
    ]
    for fname, title, ys, ylabel in specs:
        ep, y = _valid(epochs, ys)
        if not ep:
            continue
        ep, y = list(ep), list(y)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ep, y, linewidth=1.5, color='steelblue')
        _mark_baseline_updates(ax, epochs, bl_updated)
        ax.set_title(f'Rung {rung} — {title}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        p = out_dir / fname
        fig.tight_layout()
        fig.savefig(p, dpi=180)
        plt.close(fig)
        created.append(p)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Rung {rung} — Training overview', fontsize=13)
    panel_data = [
        (axes[0, 0], 'Validation cost',       val_costs,  'Cost',      'steelblue'),
        (axes[0, 1], 'Mean training reward',  rewards,    'Reward',    'darkorange'),
        (axes[1, 0], 'Actor loss',            losses,     'Loss',      'crimson'),
        (axes[1, 1], 'Gradient norm (mean)',  grad_means, 'Grad norm', 'purple'),
    ]
    for ax, title, ys, ylabel, color in panel_data:
        ep, y = _valid(epochs, ys)
        ep, y = list(ep), list(y)
        if ep:
            ax.plot(ep, y, linewidth=1.5, color=color)
            _mark_baseline_updates(ax, epochs, bl_updated)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    ep, y = _valid(epochs, grad_maxes)
    ep, y = list(ep), list(y)
    if ep:
        axes[1, 1].plot(ep, y, linewidth=0.8, color='plum', linestyle='--', label='max')
        axes[1, 1].legend(fontsize=8)

    from matplotlib.lines import Line2D
    legend_el = Line2D([0], [0], color='green', alpha=0.5, linestyle='--', linewidth=0.8)
    fig.legend([legend_el], ['Baseline updated'], loc='lower right', fontsize=8)

    overview_path = out_dir / 'training_overview.png'
    fig.tight_layout()
    fig.savefig(overview_path, dpi=180)
    plt.close(fig)
    created.append(overview_path)

    print(f'  Training curves: {len(created)} plots -> {out_dir}')
    return created

def plot_generalization(
    gen_path: Path,
    out_dir: Path,
    rung: str,
) -> Optional[Path]:
    rows = _read_csv(gen_path)
    if not rows:
        print(f'  [SKIP] No generalization CSV: {gen_path}')
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    by_size: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        try:
            n = int(float(r['n_req_test']))
            c = float(r['total_per_req_mean'])
            by_size[n].append(c)
        except (KeyError, ValueError):
            continue

    if not by_size:
        return None

    n_vals = sorted(by_size)
    means  = [np.mean(by_size[n]) for n in n_vals]
    stds   = [np.std(by_size[n])  for n in n_vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(n_vals, means, yerr=stds, marker='o', linewidth=1.5,
                capsize=4, color='steelblue', label='RL (HetGAT)')

    rung_sizes = {'A': 5, 'B': 10, 'C': 25, 'D': 60}
    train_n = rung_sizes.get(rung)
    if train_n in n_vals:
        ax.axvline(train_n, color='green', linestyle='--', alpha=0.6, label=f'Train size (n={train_n})')

    ax.set_title(f'Rung {rung} — Generalization (cost/req vs problem size)')
    ax.set_xlabel('Number of requests (n_req)')
    ax.set_ylabel('Cost per request')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend()

    out_path = out_dir / 'generalization.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'  Generalization plot: {out_path}')
    return out_path

def plot_ablation(
    abl_path: Path,
    out_dir: Path,
    rung: str,
) -> Optional[Path]:
    rows = _read_csv(abl_path)
    if not rows:
        print(f'  [SKIP] No ablation CSV: {abl_path}')
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    configs  = [r.get('config', '') for r in rows]
    costs    = _col(rows, 'total_per_req_mean')
    stds     = _col(rows, 'total_per_req_std')

    order = sorted(range(len(configs)), key=lambda i: (configs[i] != 'full', costs[i]))
    configs = [configs[i] for i in order]
    costs   = [costs[i]   for i in order]
    stds    = [stds[i]    for i in order]

    baseline_cost = costs[0] if configs[0] == 'full' else min(costs)
    colors = ['#2ca02c' if c == 'full' else
              ('#d62728' if costs[i] > baseline_cost * 1.05 else '#1f77b4')
              for i, c in enumerate(configs)]

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = range(len(configs))
    ax.bar(xs, costs, yerr=stds, capsize=3, color=colors)
    ax.axhline(baseline_cost, color='#2ca02c', linestyle='--', alpha=0.6, label='Full model')
    ax.set_title(f'Rung {rung} — Edge-feature ablation (cost/req)')
    ax.set_ylabel('Cost per request')
    ax.set_xticks(list(xs))
    ax.set_xticklabels(configs, rotation=35, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    out_path = out_dir / 'ablation.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'  Ablation plot: {out_path}')
    return out_path

def main() -> None:
    parser = argparse.ArgumentParser(description='Plot training curves, generalization, ablation')
    parser.add_argument('--rung', default='A', choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--logs-dir', default='logs',
                        help='Directory containing training_{RUNG}.csv')
    parser.add_argument('--eval-dir', default='eval_results',
                        help='Directory containing rung{RUNG}_*.csv from evaluate.py')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory for plots (default: logs/plots_{RUNG})')
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.logs_dir) / f'plots_{args.rung}'
    logs_dir = Path(args.logs_dir)
    eval_dir = Path(args.eval_dir)

    print(f'Plotting Rung {args.rung} -> {out_dir}')

    log_csv   = logs_dir / f'training_{args.rung}.csv'
    gen_csv   = eval_dir / f'rung{args.rung}_generalization.csv'
    abl_csv   = eval_dir / f'rung{args.rung}_edge_ablation.csv'

    created = []
    created += plot_training_curves(log_csv, out_dir, args.rung)
    p = plot_generalization(gen_csv, out_dir, args.rung)
    if p:
        created.append(p)
    p = plot_ablation(abl_csv, out_dir, args.rung)
    if p:
        created.append(p)

    print(f'\nTotal: {len(created)} plots saved to {out_dir}')
    for p in created:
        print(f'  {p}')

if __name__ == '__main__':
    main()
