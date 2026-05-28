from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DEMAND_MAX = 6.0
_RUNG_CONFIG = {
    'A': dict(n_req=5,  n_uav=2,  n_adr=2, n_depots_uav=1, n_depots_adr=1),
    'B': dict(n_req=10, n_uav=4,  n_adr=3, n_depots_uav=1, n_depots_adr=1),
    'C': dict(n_req=25, n_uav=5,  n_adr=4, n_depots_uav=2, n_depots_adr=2),
    'D': dict(n_req=60, n_uav=10, n_adr=8, n_depots_uav=3, n_depots_adr=3),
}

_FEAT_X        = 0
_FEAT_Y        = 1
_FEAT_DEMAND   = 2
_FEAT_ACC_UAV  = 8
_FEAT_TW_SLACK = 10

def _load_actor(checkpoint: str):
    from VRP_Actor import Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(
        input_node_dim=11, hidden_node_dim=128,
        input_edge_dim=4,  hidden_edge_dim=16,
        conv_layers=3,
    ).to(device)
    raw   = torch.load(checkpoint, map_location=device, weights_only=False)
    state = raw.get('model_state_dict', raw)
    model.load_state_dict(state)
    model.eval()
    print(f'Loaded actor from {checkpoint}  (device={device})')
    return model, device

def collect_assignments(
    actor,
    device,
    cfg: dict,
    n_instances: int,
    seed: int = 0,
) -> List[Dict]:
    from creat_vrp import create_instance

    n_req        = cfg['n_req']
    n_uav        = cfg['n_uav']
    n_adr        = cfg['n_adr']
    n_depots_uav = cfg['n_depots_uav']
    n_depots_adr = cfg['n_depots_adr']
    n_depots     = n_depots_uav + n_depots_adr
    n_agents     = n_uav + n_adr

    records: List[Dict] = []

    for ep in range(n_instances):
        rng  = np.random.default_rng(seed + ep)
        inst = create_instance(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr, rng)

        batch = {
            k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inst.items()
        }

        with torch.no_grad():
            tour, _, _ = actor(batch, n_uav, n_adr, greedy=True,
                               checkpoint_encoder=False, training=False)

        tour_np = tour[0].cpu().numpy()

        node_to_mode: Dict[int, str] = {}
        for agent_idx in range(n_agents):
            mode = 'uav' if agent_idx < n_uav else 'adr'
            for node in tour_np[agent_idx]:
                node = int(node)
                if n_depots <= node < n_depots + n_req:
                    node_to_mode[node] = mode

        x        = inst['x'].numpy()
        coords   = x[:, _FEAT_X:_FEAT_Y + 1]
        demand   = x[:, _FEAT_DEMAND] * DEMAND_MAX
        tw_slack = x[:, _FEAT_TW_SLACK]
        acc_uav  = x[:, _FEAT_ACC_UAV]

        depot_coords = coords[:n_depots]

        inacc_mask = acc_uav == 0.0
        inacc_nodes = np.where(inacc_mask)[0]

        wind_mag = float(inst['wind'][0]) if 'wind' in inst else 0.0

        for req_idx in range(n_req):
            pickup_node = n_depots + req_idx
            mode = node_to_mode.get(pickup_node)
            if mode is None:
                continue

            pos = coords[pickup_node]

            dist = float(np.min(np.linalg.norm(pos - depot_coords, axis=1)))

            if len(inacc_nodes) > 0:
                nofly_dist = float(np.min(np.linalg.norm(pos - coords[inacc_nodes], axis=1)))
            else:
                nofly_dist = 1.0

            records.append({
                'demand':    float(demand[pickup_node]),
                'dist':      dist,
                'tw_slack':  float(tw_slack[pickup_node]),
                'nofly_dist': nofly_dist,
                'mode':      mode,
                'wind_mag':  wind_mag,
            })

        if (ep + 1) % 50 == 0:
            print(f'  {ep + 1}/{n_instances} instances  '
                  f'({len(records)} records, '
                  f'UAV {sum(1 for r in records if r["mode"]=="uav")/max(len(records),1):.0%})')

    return records

def _bin_records(
    records: List[Dict],
    x_key: str,
    y_key: str,
    x_bins: int = 7,
    y_bins: int = 7,
    x_range: Optional[Tuple] = None,
    y_range: Optional[Tuple] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xv = np.array([r[x_key] for r in records])
    yv = np.array([r[y_key] for r in records])
    is_uav = np.array([1.0 if r['mode'] == 'uav' else 0.0 for r in records])

    x_range = x_range or (float(xv.min()), float(xv.max()))
    y_range = y_range or (float(yv.min()), float(yv.max()))

    x_edges = np.linspace(x_range[0], x_range[1], x_bins + 1)
    y_edges = np.linspace(y_range[0], y_range[1], y_bins + 1)

    uav_sum   = np.zeros((y_bins, x_bins))
    total_sum = np.zeros((y_bins, x_bins))

    for xi, yi, u in zip(xv, yv, is_uav):
        cx = min(int((xi - x_range[0]) / max(x_range[1] - x_range[0], 1e-9) * x_bins), x_bins - 1)
        cy = min(int((yi - y_range[0]) / max(y_range[1] - y_range[0], 1e-9) * y_bins), y_bins - 1)
        if cx < 0 or cy < 0:
            continue
        uav_sum[cy, cx]   += u
        total_sum[cy, cx] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        frac  = np.where(total_sum > 0, uav_sum / total_sum, np.nan)
        p_sm  = (uav_sum + 0.5) / (total_sum + 1.0)
        logit = np.where(total_sum >= 3, np.log(p_sm / (1.0 - p_sm)), np.nan)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    return frac, logit, x_centers, y_centers, total_sum

def _heatmap_ax(ax, grid, x_centers, y_centers, counts,
                xlabel, ylabel, title,
                cmap='RdBu_r', vmin=-2.5, vmax=2.5,
                cbar_label='Mode-preference logit Δᵢ\n(+= UAV preferred, −= ADR preferred)'):
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(
        masked, origin='lower', aspect='auto', cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
    )
    ny, nx = grid.shape
    for cy in range(ny):
        for cx in range(nx):
            n = int(counts[cy, cx])
            if n < 3 or np.isnan(grid[cy, cx]):
                continue
            val = grid[cy, cx]
            txt = f'{val:+.1f}' if abs(vmax) > 1.5 else f'{val:.0%}'
            ax.text(
                x_centers[cx], y_centers[cy], txt,
                ha='center', va='center', fontsize=6.5,
                color='white' if abs(val) > abs(vmax) * 0.55 else 'black',
            )
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    return im

def plot_mode_logit_demand_dist(records: List[Dict], out_dir: Path) -> Optional[Path]:
    if len(records) < 20:
        return None
    frac, logit, xc, yc, counts = _bin_records(
        records, 'demand', 'dist',
        x_bins=6, y_bins=6,
        x_range=(1.0, DEMAND_MAX),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im0 = _heatmap_ax(
        axes[0], logit, xc, yc, counts,
        xlabel='Request demand (kg)',
        ylabel='Distance from nearest depot (norm.)',
        title='Mode-preference logit Δᵢ\n(demand × distance)',
        cmap='RdBu_r', vmin=-2.5, vmax=2.5,
    )
    plt.colorbar(im0, ax=axes[0], label='logit(UAV frac)  (+= UAV, −= ADR)')

    axes[0].text(1.3, yc[-1] * 0.88, 'far/light\n→ UAV', fontsize=8,
                 color='darkblue', ha='left', style='italic')
    axes[0].text(5.0, yc[0]  * 1.15, 'near/heavy\n→ ADR', fontsize=8,
                 color='darkred', ha='right', style='italic')

    im1 = _heatmap_ax(
        axes[1], frac, xc, yc, counts,
        xlabel='Request demand (kg)',
        ylabel='Distance from nearest depot (norm.)',
        title='UAV assignment fraction\n(same data, raw probability)',
        cmap='RdYlBu_r', vmin=0.0, vmax=1.0,
        cbar_label='UAV fraction',
    )
    plt.colorbar(im1, ax=axes[1], label='UAV assignment fraction')

    fig.suptitle(
        f'Claim 2: Policy learned mode split  (n={len(records):,} requests)',
        fontsize=11, y=1.01,
    )
    p = out_dir / 'mode_logit_demand_dist.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return p

def plot_mode_logit_urgency_nofly(records: List[Dict], out_dir: Path) -> Optional[Path]:
    if len(records) < 20:
        return None

    nofly_vals = np.array([r['nofly_dist'] for r in records])
    if nofly_vals.std() < 1e-4:
        print('  [SKIP] mode_logit_urgency_nofly: no no-fly zone variation in instances')
        return None

    frac, logit, xc, yc, counts = _bin_records(
        records, 'tw_slack', 'nofly_dist',
        x_bins=6, y_bins=6,
        x_range=(0.0, 1.0),
        y_range=(0.0, float(np.percentile(nofly_vals, 95))),
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = _heatmap_ax(
        ax, logit, xc, yc, counts,
        xlabel='Time-window slack (0 = urgent, 1 = loose)',
        ylabel='Distance to nearest no-fly zone (norm.)',
        title='Mode-preference logit Δᵢ: urgency × no-fly proximity\n'
              'Bottom = inside/near restricted zone (ADR forced)\n'
              'Top = far from restrictions',
        cmap='RdBu_r', vmin=-2.5, vmax=2.5,
    )
    plt.colorbar(im, ax=ax, label='logit(UAV frac)  (+= UAV preferred, −= ADR preferred)')

    ax.text(0.05, float(yc[-1]) * 0.88, 'urgent + no restriction\n→ UAV', fontsize=8,
            color='darkblue', ha='left', style='italic')
    ax.text(0.85, float(yc[0])  * 1.1,  'no-fly zone\n→ ADR only', fontsize=8,
            color='darkred', ha='right', style='italic')

    p = out_dir / 'mode_logit_urgency_nofly.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def plot_mode_wind_panels(records: List[Dict], out_dir: Path) -> Optional[Path]:
    if len(records) < 30:
        return None

    wind_bins = [
        ('Low wind\n(0–4 m/s)',    0.0,  4.0),
        ('Medium wind\n(4–8 m/s)', 4.0,  8.0),
        ('High wind\n(8–12 m/s)', 8.0, 12.1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    last_im, any_ok = None, False

    for ax, (label, lo, hi) in zip(axes, wind_bins):
        subset = [r for r in records if lo <= r['wind_mag'] < hi]
        if len(subset) < 10:
            ax.set_title(f'{label}\n(n={len(subset)}, insufficient data)')
            ax.axis('off')
            continue

        _, logit, xc, yc, counts = _bin_records(
            subset, 'demand', 'dist',
            x_bins=5, y_bins=5,
            x_range=(1.0, DEMAND_MAX),
        )
        last_im = _heatmap_ax(
            ax, logit, xc, yc, counts,
            xlabel='Demand (kg)',
            ylabel='Dist to depot (norm.)',
            title=f'{label}\n(n={len(subset):,})',
            cmap='RdBu_r', vmin=-2.5, vmax=2.5,
        )
        any_ok = True

    if not any_ok:
        plt.close(fig)
        return None

    if last_im is not None:
        plt.colorbar(last_im, ax=axes, label='logit(UAV frac)', shrink=0.8)

    fig.suptitle(
        'Mode-preference logit under different wind regimes\n'
        'Red = UAV preferred  |  Blue = ADR preferred\n'
        'Headwind increases UAV energy cost → diagonal shifts toward ADR',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    p = out_dir / 'mode_wind_panels.png'
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def plot_mode_quartile_grid(records: List[Dict], out_dir: Path) -> Optional[Path]:
    if len(records) < 16:
        return None

    demands = np.array([r['demand'] for r in records])
    dists   = np.array([r['dist']   for r in records])
    modes   = np.array([r['mode']   for r in records])

    d_med   = float(np.median(demands))
    dist_med= float(np.median(dists))

    cells = {
        (0, 0): ('Light demand\nNear depot',  (demands <= d_med) & (dists <= dist_med)),
        (0, 1): ('Light demand\nFar from depot', (demands <= d_med) & (dists > dist_med)),
        (1, 0): ('Heavy demand\nNear depot',  (demands > d_med)  & (dists <= dist_med)),
        (1, 1): ('Heavy demand\nFar from depot', (demands > d_med)  & (dists > dist_med)),
    }

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    for (row, col), (title, mask) in cells.items():
        ax   = axes[row][col]
        sub  = modes[mask]
        n    = len(sub)
        if n == 0:
            ax.set_title(title + '\n(no data)')
            ax.axis('off')
            continue

        uav_frac = (sub == 'uav').mean()
        adr_frac = (sub == 'adr').mean()

        bars = ax.bar(['UAV', 'ADR'], [uav_frac, adr_frac],
                      color=['steelblue', 'darkorange'])
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.set_title(f'{title}\n(n={n})', fontsize=9)
        ax.set_ylabel('Served fraction', fontsize=8)
        ax.axhline(0.5, color='grey', linewidth=0.7, linestyle='--')
        ax.grid(axis='y', alpha=0.3)

        for bar, frac in zip(bars, [uav_frac, adr_frac]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    frac + 0.03, f'{frac:.0%}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle(
        f'Per-mode delivery fraction by demand × distance quartile\n'
        f'(demand median={d_med:.1f} kg, distance median={dist_med:.3f} norm.)',
        fontsize=11,
    )
    fig.tight_layout()
    p = out_dir / 'mode_quartile_grid.png'
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def main():
    parser = argparse.ArgumentParser(description='Mode-preference heatmap analysis')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_actor.pt')
    parser.add_argument('--rung', default='C', choices=list(_RUNG_CONFIG))
    parser.add_argument('--n-instances', type=int, default=300,
                        help='>=300 for reliable heatmaps; 500+ for publication')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='eval_results/plots/mode_preference')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _RUNG_CONFIG[args.rung]
    print(f'Rung {args.rung}: n_req={cfg["n_req"]}  n_uav={cfg["n_uav"]}  n_adr={cfg["n_adr"]}')

    actor, device = _load_actor(args.checkpoint)

    print(f'Collecting assignments from {args.n_instances} instances ...')
    records = collect_assignments(actor, device, cfg, args.n_instances, seed=args.seed)
    print(f'Total served requests: {len(records)}'
          f'  UAV={sum(1 for r in records if r["mode"]=="uav")/max(len(records),1):.0%}'
          f'  ADR={sum(1 for r in records if r["mode"]=="adr")/max(len(records),1):.0%}')

    plots = [
        ('mode_logit_demand_dist.png',   lambda: plot_mode_logit_demand_dist(records, out_dir)),
        ('mode_logit_urgency_nofly.png', lambda: plot_mode_logit_urgency_nofly(records, out_dir)),
        ('mode_wind_panels.png',         lambda: plot_mode_wind_panels(records, out_dir)),
        ('mode_quartile_grid.png',       lambda: plot_mode_quartile_grid(records, out_dir)),
    ]
    for fname, fn in plots:
        try:
            p = fn()
            print(f'Saved: {p}' if p else f'SKIP {fname}: insufficient data')
        except Exception as exc:
            print(f'ERROR {fname}: {exc}')

    print('\nDone.')

if __name__ == '__main__':
    main()
