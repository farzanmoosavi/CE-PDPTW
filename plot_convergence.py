from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def _read_csv(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'CSV not found: {path}')
    with p.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

def _col(rows: List[Dict], key: str, default=float('nan')) -> np.ndarray:
    out = []
    for r in rows:
        try:
            v = float(r.get(key, default))
            out.append(v if math.isfinite(v) else float('nan'))
        except (TypeError, ValueError):
            out.append(float('nan'))
    return np.array(out)

_PALETTE = [
    '#1f77b4',
    '#d62728',
    '#2ca02c',
    '#ff7f0e',
    '#9467bd',
]

def plot_convergence(
    csv_paths: List[str],
    labels: Optional[List[str]] = None,
    out_path: str = 'convergence.png',
    show_train_reward: bool = False,
    smooth_window: int = 1,
    dpi: int = 180,
) -> None:
    if labels is None:
        labels = [Path(p).stem for p in csv_paths]
    if len(labels) < len(csv_paths):
        labels += [Path(p).stem for p in csv_paths[len(labels):]]

    n_series = len(csv_paths)
    n_panels = 2 if show_train_reward else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax_val = axes[0]
    ax_rew = axes[1] if show_train_reward else None

    best_epoch_markers = []

    for i, (path, label) in enumerate(zip(csv_paths, labels)):
        rows = _read_csv(path)
        if not rows:
            print(f'  [WARN] Empty CSV: {path}')
            continue

        epochs   = _col(rows, 'epoch')
        val_mean = _col(rows, 'val_cost')
        val_std  = _col(rows, 'val_cost_std')

        color = _PALETTE[i % len(_PALETTE)]

        if smooth_window > 1:
            def _smooth(x):
                k = np.ones(smooth_window) / smooth_window
                return np.convolve(x, k, mode='same')
            val_mean = _smooth(val_mean)
            val_std  = _smooth(val_std)

        ax_val.plot(epochs, val_mean, color=color, linewidth=2.0, label=label)

        has_std = np.any(val_std > 1e-6)
        if has_std:
            ax_val.fill_between(
                epochs,
                val_mean - val_std,
                val_mean + val_std,
                color=color, alpha=0.18,
            )

        best_idx = int(np.nanargmin(val_mean))
        best_cost = val_mean[best_idx]
        ax_val.scatter(epochs[best_idx], best_cost, marker='*', s=180,
                       color=color, zorder=5, edgecolors='black', linewidth=0.5)
        best_epoch_markers.append((epochs[best_idx], best_cost, label, color))

        if show_train_reward and ax_rew is not None:
            train_rew = _col(rows, 'train_reward_mean')
            if smooth_window > 1:
                train_rew = _smooth(train_rew)
            ax_rew.plot(epochs, train_rew, color=color, linewidth=1.5,
                        linestyle='--', label=label)

    for (ep, cost, lbl, col) in best_epoch_markers:
        ax_val.annotate(
            f'ep {int(ep)}\n{cost:.3f}',
            xy=(ep, cost), xytext=(8, -14),
            textcoords='offset points',
            fontsize=7, color=col,
            arrowprops=dict(arrowstyle='->', color=col, lw=0.8),
        )

    ax_val.set_ylabel('Validation cost (mean cost / request)', fontsize=10)
    ax_val.set_title('Training convergence — val cost per epoch\n'
                     '(★ = best, shaded band = ±1 std over val seeds)', fontsize=10)
    ax_val.legend(fontsize=9, loc='upper right')
    ax_val.grid(alpha=0.3)
    ax_val.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    if show_train_reward and ax_rew is not None:
        ax_rew.set_ylabel('Train reward (mean per step)', fontsize=10)
        ax_rew.legend(fontsize=9, loc='upper right')
        ax_rew.grid(alpha=0.3)

    axes[-1].set_xlabel('Epoch', fontsize=10)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

def main() -> None:
    parser = argparse.ArgumentParser(description='Convergence shadow-plot')
    parser.add_argument('--csv', nargs='+', required=True,
                        help='One or more training CSV files')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Display labels (one per CSV). Defaults to file stems.')
    parser.add_argument('--out', default='convergence.png',
                        help='Output PNG path')
    parser.add_argument('--show-train-reward', action='store_true',
                        help='Add a second panel with train_reward_mean')
    parser.add_argument('--smooth', type=int, default=1,
                        help='Moving-average window for smoothing (1 = no smoothing)')
    parser.add_argument('--dpi', type=int, default=180)
    args = parser.parse_args()

    plot_convergence(
        csv_paths=args.csv,
        labels=args.labels,
        out_path=args.out,
        show_train_reward=args.show_train_reward,
        smooth_window=args.smooth,
        dpi=args.dpi,
    )

if __name__ == '__main__':
    main()
