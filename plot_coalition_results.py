from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        print(f'  [SKIP] {path} not found')
        return []
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

def _f(row: Dict, key: str, default=float('nan')) -> float:
    try:
        v = float(row.get(key, default))
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default

def _s(row: Dict, key: str, default='') -> str:
    return str(row.get(key, default))

def plot_gain_by_scenario(rows: List[Dict], out_dir: Path) -> Optional[Path]:
    if not rows:
        return None
    rows_s = sorted(rows, key=lambda r: _f(r, 'coalition_gain_mean'), reverse=True)[:30]
    labels = [_s(r, 'scenario') for r in rows_s]
    gains  = [_f(r, 'coalition_gain_mean') for r in rows_s]
    stds   = [_f(r, 'coalition_gain_std', 0) for r in rows_s]
    core   = [_f(r, 'core_rate_shapley', 0) for r in rows_s]

    cmap = plt.cm.RdYlGn
    colors = [cmap(min(max(c, 0), 1)) for c in core]

    fig, ax = plt.subplots(figsize=(14, 5))
    xs = range(len(labels))
    bars = ax.bar(xs, gains, yerr=stds, capsize=2, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Top 30 scenarios by coalition gain\n(colour = core rate: green=high, red=low)')
    ax.set_ylabel('Coalition gain (UAV + ADR standalone - coalition cost)')
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=6)
    ax.grid(axis='y', alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label='Core rate (Shapley)')

    p = out_dir / 'gain_by_scenario.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def plot_core_rate_heatmap(rows: List[Dict], out_dir: Path) -> Optional[Path]:
    if not rows:
        return None

    market_fracs = sorted({_f(r, 'uav_market_frac') for r in rows if not math.isnan(_f(r, 'uav_market_frac'))})
    depot_vals   = [0, 1]

    data = np.full((len(depot_vals), len(market_fracs)), float('nan'))
    counts = np.zeros_like(data)
    for r in rows:
        mf = _f(r, 'uav_market_frac')
        ds = int(_f(r, 'depot_shared', 0))
        cr = _f(r, 'core_rate_shapley')
        if math.isnan(mf) or math.isnan(cr):
            continue
        try:
            mi = market_fracs.index(mf)
        except ValueError:
            continue
        di = depot_vals.index(ds)
        if math.isnan(data[di, mi]):
            data[di, mi] = 0.0
        data[di, mi] = (data[di, mi] * counts[di, mi] + cr) / (counts[di, mi] + 1)
        counts[di, mi] += 1

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(data, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(market_fracs)))
    ax.set_xticklabels([f'UAV={int(m*100)}%' for m in market_fracs])
    ax.set_yticks(range(len(depot_vals)))
    ax.set_yticklabels(['Sep. depots', 'Shared depots'])
    ax.set_title('Core rate (Shapley) by market split x depot config')
    plt.colorbar(im, ax=ax, label='Core rate (both firms benefit)')

    for di in range(len(depot_vals)):
        for mi in range(len(market_fracs)):
            v = data[di, mi]
            if not math.isnan(v):
                ax.text(mi, di, f'{v:.1%}', ha='center', va='center', fontsize=9,
                        color='white' if v < 0.5 else 'black')

    p = out_dir / 'core_rate_heatmap.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def plot_benefit_by_method(rows: List[Dict], out_dir: Path) -> Optional[Path]:
    if not rows:
        return None

    methods = ['shapley', 'epm', 'pam']
    uav_means = [np.nanmean([_f(r, f'uav_gain_mean_{m}') for r in rows]) for m in methods]
    adr_means = [np.nanmean([_f(r, f'adr_gain_mean_{m}') for r in rows]) for m in methods]
    uav_p5s   = [np.nanmean([_f(r, f'uav_gain_p5_{m}')   for r in rows]) for m in methods]
    adr_p5s   = [np.nanmean([_f(r, f'adr_gain_p5_{m}')   for r in rows]) for m in methods]
    core_rates= [np.nanmean([_f(r, f'core_rate_{m}')      for r in rows]) for m in methods]

    x = np.arange(len(methods))
    width = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(x - width/2, uav_means, width, label='UAV firm', color='steelblue')
    ax1.bar(x + width/2, adr_means, width, label='ADR firm', color='darkorange')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_title('Mean per-firm cost saving\n(positive = firm pays less in coalition)')
    ax1.set_ylabel('Mean gain')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in methods])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x - width/2, uav_p5s, width, label='UAV firm p5', color='steelblue', alpha=0.7)
    ax2.bar(x + width/2, adr_p5s, width, label='ADR firm p5', color='darkorange', alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    for i, cr in enumerate(core_rates):
        ax2.text(i, max(uav_p5s[i], adr_p5s[i]) + 0.5, f'core={cr:.0%}',
                 ha='center', fontsize=8, color='darkgreen')
    ax2.set_title('5th-pct per-firm gain (worst-case, 95% CI)\nnegative = firm may lose in 5% of episodes')
    ax2.set_ylabel('Gain (p5)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in methods])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    p = out_dir / 'benefit_by_method.png'
    fig.suptitle('Per-firm benefit distribution across all scenarios')
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def plot_conviction_bar(rows: List[Dict], out_dir: Path) -> Optional[Path]:
    if not rows:
        return None
    rows_s = sorted(rows, key=lambda r: _f(r, 'conviction', 0), reverse=True)[:20]
    labels = [_s(r, 'scenario') for r in rows_s]
    convs  = [_f(r, 'conviction', 0) for r in rows_s]
    cores  = [_f(r, 'core_rate_shapley', 0) for r in rows_s]
    gains  = [_f(r, 'coalition_gain_mean', 0) for r in rows_s]

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ['#2ca02c' if c >= 0.7 else ('#ff7f0e' if c >= 0.4 else '#d62728') for c in convs]
    ax.bar(range(len(labels)), convs, color=colors)
    ax.set_title('Top 20 scenarios by conviction score\n'
                 '(green >= 0.7 = strong case, orange = moderate, red = weak)')
    ax.set_ylabel('Conviction score [0-1]')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=6.5)
    ax.axhline(0.7, color='green',  linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(0.4, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    for i, (c, g) in enumerate(zip(cores, gains)):
        ax.text(i, convs[i] + 0.02, f'{c:.0%}\n{g:+.1f}',
                ha='center', va='bottom', fontsize=5.5, color='black')

    p = out_dir / 'conviction_bar.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def plot_factor_sensitivity(rows: List[Dict], out_dir: Path) -> Optional[Path]:
    if len(rows) < 4:
        return None
    from scipy.stats import pearsonr

    gains = np.array([_f(r, 'coalition_gain_mean') for r in rows])
    valid = ~np.isnan(gains)
    gains = gains[valid]
    filtered = [r for r, v in zip(rows, valid) if v]

    factors = {
        'UAV market frac':   [_f(r, 'uav_market_frac')   for r in filtered],
        'eps_shared':        [_f(r, 'eps_shared')         for r in filtered],
        'Depot shared':      [_f(r, 'depot_shared')       for r in filtered],
        'alpha_asymmetry':   [_f(r, 'alpha_asymmetry')    for r in filtered],
        'Charging cost UAV': [_f(r, 'charging_cost_uav')  for r in filtered],
        'Charging cost ADR': [_f(r, 'charging_cost_adr')  for r in filtered],
    }

    results = []
    for fname, fvals in factors.items():
        fv = np.array(fvals)
        if fv.std() < 1e-9 or np.any(np.isnan(fv)):
            continue
        r_val, p_val = pearsonr(fv, gains)
        results.append((fname, r_val, p_val))

    if not results:
        return None

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    names  = [x[0] for x in results]
    rvals  = [x[1] for x in results]
    pvals  = [x[2] for x in results]

    colors = ['#2ca02c' if r > 0 else '#d62728' for r in rvals]
    alphas = [1.0 if p < 0.05 else 0.4 for p in pvals]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, [abs(r) for r in rvals], color=colors)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    ax.axvline(0, color='black', linewidth=0.6)

    for i, (r, p) in enumerate(zip(rvals, pvals)):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        direction = '+' if r > 0 else '-'
        ax.text(abs(r) + 0.005, i, f'{direction}  r={r:+.3f} {sig}',
                va='center', fontsize=8)

    ax.set_xlabel('|Pearson r| with coalition_gain_mean')
    ax.set_title('Factor sensitivity\n(green=positive driver, red=negative; faded=not significant)')
    ax.set_xlim(0, 1.25)
    ax.grid(axis='x', alpha=0.3)

    p = out_dir / 'factor_sensitivity.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def plot_gain_distribution(rows: List[Dict], out_dir: Path) -> Optional[Path]:
    if not rows:
        return None
    gains = [_f(r, 'coalition_gain_mean') for r in rows if not math.isnan(_f(r, 'coalition_gain_mean'))]
    if not gains:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gains, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0,            color='black',  linewidth=1.2, linestyle='--', label='Breakeven')
    ax.axvline(np.mean(gains), color='darkorange', linewidth=1.5, linestyle='-',
               label=f'Mean={np.mean(gains):.2f}')
    pct_positive = 100 * np.mean(np.array(gains) >= 0)
    ax.set_title(f'Coalition gain distribution across {len(gains)} scenarios\n'
                 f'({pct_positive:.0f}% of scenarios have positive gain)')
    ax.set_xlabel('Coalition gain (positive = coalition cheaper than standalone sum)')
    ax.set_ylabel('Number of scenarios')
    ax.legend()
    ax.grid(alpha=0.3)

    p = out_dir / 'gain_distribution.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def plot_participation_rate_bar(rows: List[Dict], out_dir: Path) -> Optional[Path]:
    if not rows:
        return None

    methods = ['shapley', 'epm', 'pam']

    def _avg(key):
        vals = [_f(r, key) for r in rows if not math.isnan(_f(r, key))]
        return float(np.nanmean(vals)) if vals else float('nan')

    uav_rates  = [_avg(f'uav_ir_rate_{m}')  for m in methods]
    adr_rates  = [_avg(f'adr_ir_rate_{m}')  for m in methods]
    core_rates = [_avg(f'core_rate_{m}')    for m in methods]

    x = np.arange(len(methods))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w,    uav_rates,  w, label='UAV firm',      color='steelblue')
    ax.bar(x,        adr_rates,  w, label='ADR firm',      color='darkorange')
    ax.bar(x + w,    core_rates, w, label='Both firms',    color='#2ca02c')

    ax.axhline(0.5, color='grey', linewidth=0.8, linestyle='--', alpha=0.6,
               label='50% threshold')
    ax.axhline(0.7, color='green', linewidth=0.8, linestyle=':', alpha=0.6,
               label='70% (strong case)')

    for i, (u, a, c) in enumerate(zip(uav_rates, adr_rates, core_rates)):
        for xi, val, clr in [(i - w, u, 'steelblue'),
                             (i,     a, 'darkorange'),
                             (i + w, c, '#2ca02c')]:
            if not math.isnan(val):
                ax.text(xi, val + 0.012, f'{val:.0%}',
                        ha='center', va='bottom', fontsize=8, color=clr)

    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_xlabel('Allocation method')
    ax.set_ylabel('Participation rate\n(fraction of episodes firm benefits)')
    ax.set_title('Coalition participation rate by allocation method\n'
                 '(fraction of episodes each firm voluntarily enters the coalition)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    p = out_dir / 'participation_rate_bar.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def plot_gain_contour(rows: List[Dict], out_dir: Path) -> Optional[Path]:
    if not rows:
        return None

    from scipy.interpolate import griddata

    pts, vals = [], []
    for r in rows:
        x = _f(r, 'uav_market_frac')
        y = _f(r, 'eps_shared')
        z = _f(r, 'coalition_gain_mean')
        if any(math.isnan(v) for v in (x, y, z)):
            continue
        pts.append((x, y))
        vals.append(z)

    if len(pts) < 4:
        return None

    pts_arr = np.array(pts)
    vals_arr = np.array(vals)

    xi = np.linspace(pts_arr[:, 0].min(), pts_arr[:, 0].max(), 60)
    yi = np.linspace(pts_arr[:, 1].min(), pts_arr[:, 1].max(), 60)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata(pts_arr, vals_arr, (Xi, Yi), method='linear')

    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(Xi * 100, Yi * 100, Zi, levels=12, cmap='RdYlGn')
    cs = ax.contour(Xi * 100, Yi * 100, Zi, levels=12, colors='k', linewidths=0.4, alpha=0.4)
    ax.clabel(cs, fmt='%.1f', fontsize=7)
    plt.colorbar(cf, ax=ax, label='Mean coalition gain')
    ax.set_xlabel('UAV market share (%)')
    ax.set_ylabel('Cross-mode sharing tolerance ε_shared (%)')
    ax.set_title('Coalition gain over (market share × ε_shared)\n'
                 'Green = higher gain; contour lines show iso-gain')
    ax.scatter(pts_arr[:, 0] * 100, pts_arr[:, 1] * 100,
               c=vals_arr, cmap='RdYlGn', s=30, edgecolors='k', linewidths=0.5, zorder=5)

    p = out_dir / 'gain_contour.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def plot_operational_table(rows: List[Dict], out_dir: Path) -> Optional[Path]:
    if not rows:
        return None

    def _avg(key):
        vals = [_f(r, key) for r in rows if not math.isnan(_f(r, key))]
        return float(np.mean(vals)) if vals else float('nan')

    n_req_avg = _avg('n_req') or 1.0

    uav_cpr  = _avg('uav_cost_per_req')
    adr_cpr  = _avg('adr_cost_per_req')
    coal_cpr = _avg('coal_cost_per_req')

    uav_sh_cpr  = uav_cpr  - _avg('uav_gain_mean_shapley')  / max(n_req_avg, 1)
    adr_sh_cpr  = adr_cpr  - _avg('adr_gain_mean_shapley')  / max(n_req_avg, 1)

    uav_ontime  = 1.0 - _avg('uav_late_pct_mean')
    adr_ontime  = 1.0 - _avg('adr_late_pct_mean')
    coal_ontime = 1.0 - _avg('coal_late_pct_mean')

    uav_served  = _avg('uav_served_frac_mean')
    adr_served  = _avg('adr_served_frac_mean')
    coal_served = _avg('coal_served_frac_mean')

    uav_undeliv  = 1.0 - uav_served
    adr_undeliv  = 1.0 - adr_served
    coal_undeliv = 1.0 - coal_served

    col_labels = ['Cost/req\n(standalone)', 'Cost/req\n(Shapley)', 'On-time %', 'Served %', 'Undelivered %']
    row_labels = ['UAV firm', 'ADR firm', 'Coalition (joint)']

    def _fmt(v, pct=False):
        if math.isnan(v):
            return 'N/A'
        return f'{v:.1%}' if pct else f'{v:.3f}'

    table_data = [
        [_fmt(uav_cpr), _fmt(uav_sh_cpr), _fmt(uav_ontime, True), _fmt(uav_served, True), _fmt(uav_undeliv, True)],
        [_fmt(adr_cpr), _fmt(adr_sh_cpr), _fmt(adr_ontime, True), _fmt(adr_served, True), _fmt(adr_undeliv, True)],
        [_fmt(coal_cpr), '—', _fmt(coal_ontime, True), _fmt(coal_served, True), _fmt(coal_undeliv, True)],
    ]

    row_colors = [['#ddeeff'] * 5, ['#ffeedd'] * 5, ['#ddffdd'] * 5]

    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.axis('off')
    tbl = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    ax.set_title('Per-firm operational breakdown (averaged over all scenarios, Shapley allocation)',
                 fontsize=11, pad=14)

    p = out_dir / 'operational_table.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return p

def plot_ontime_comparison(rows: List[Dict], out_dir: Path) -> Optional[Path]:
    if not rows:
        return None

    buckets = {}
    for r in rows:
        mf  = _f(r, 'uav_market_frac')
        key = f'UAV {int(round(mf * 100))}%' if not math.isnan(mf) else 'unknown'
        buckets.setdefault(key, {'uav': [], 'adr': [], 'coal': []})
        u = _f(r, 'uav_late_pct_mean')
        a = _f(r, 'adr_late_pct_mean')
        c = _f(r, 'coal_late_pct_mean')
        if not math.isnan(u): buckets[key]['uav'].append(1.0 - u)
        if not math.isnan(a): buckets[key]['adr'].append(1.0 - a)
        if not math.isnan(c): buckets[key]['coal'].append(1.0 - c)

    if not buckets:
        return None

    labels   = sorted(buckets)
    uav_vals = [np.nanmean(buckets[k]['uav'])  for k in labels]
    adr_vals = [np.nanmean(buckets[k]['adr'])  for k in labels]
    coal_vals= [np.nanmean(buckets[k]['coal']) for k in labels]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, uav_vals,  w, label='UAV standalone', color='steelblue')
    ax.bar(x,     adr_vals,  w, label='ADR standalone', color='darkorange')
    ax.bar(x + w, coal_vals, w, label='Joint coalition', color='#2ca02c')

    for i in range(len(labels)):
        best_standalone = max(uav_vals[i], adr_vals[i])
        delta = coal_vals[i] - best_standalone
        if not math.isnan(delta):
            color = 'darkgreen' if delta >= 0 else 'red'
            ax.text(x[i] + w, coal_vals[i] + 0.005, f'{delta:+.1%}',
                    ha='center', va='bottom', fontsize=8, color=color)

    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('On-time delivery rate')
    ax.set_title('Service level: on-time delivery rate by market split\n'
                 '(Δ annotation = coalition vs best standalone)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    p = out_dir / 'ontime_comparison.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def plot_rl_vs_greedy(greedy_rows: List[Dict], rl_rows: List[Dict], out_dir: Path) -> Optional[Path]:
    if not greedy_rows or not rl_rows or len(greedy_rows) != len(rl_rows):
        return None

    metrics = [
        ('coalition_gain_mean',  'Coalition gain'),
        ('core_rate_shapley',    'Core rate (Shapley)'),
        ('uav_ir_rate_shapley',  'UAV IR rate'),
        ('adr_ir_rate_shapley',  'ADR IR rate'),
    ]

    g_means = {k: np.nanmean([_f(r, k) for r in greedy_rows]) for k, _ in metrics}
    r_means = {k: np.nanmean([_f(r, k) for r in rl_rows])     for k, _ in metrics}

    labels = [lab for _, lab in metrics]
    g_vals = [g_means[k] for k, _ in metrics]
    r_vals = [r_means[k] for k, _ in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width/2, g_vals, width, label='Greedy', color='steelblue', alpha=0.85)
    ax.bar(x + width/2, r_vals, width, label='RL',     color='#2ca02c',   alpha=0.85)

    for i, (gv, rv) in enumerate(zip(g_vals, r_vals)):
        delta = rv - gv
        color = 'darkgreen' if delta >= 0 else 'red'
        ax.text(x[i] + width/2, rv + 0.01,
                f'{delta:+.3f}', ha='center', va='bottom', fontsize=8, color=color)

    ax.set_title('RL vs GreedyInsertion — coalition performance comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    p = out_dir / 'rl_vs_greedy.png'
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p

def main() -> None:
    parser = argparse.ArgumentParser(description='Plot coalition sweep results')
    parser.add_argument('--eval-dir', default='eval_results')
    parser.add_argument('--out-dir',  default=None,
                        help='Output directory (default: eval_results/plots/coalition)')
    parser.add_argument('--greedy-csv', default=None,
                        help='Path to coalition_sweep_greedy.csv (default: auto)')
    parser.add_argument('--rl-csv', default=None,
                        help='Path to coalition_sweep_rl.csv for RL vs Greedy plot')
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.out_dir) if args.out_dir else eval_dir / 'plots' / 'coalition'
    out_dir.mkdir(parents=True, exist_ok=True)

    greedy_csv = Path(args.greedy_csv) if args.greedy_csv else eval_dir / 'coalition_sweep_greedy.csv'
    rl_csv     = Path(args.rl_csv)     if args.rl_csv     else eval_dir / 'coalition_sweep_rl.csv'

    greedy_rows = _read_csv(greedy_csv)
    rl_rows     = _read_csv(rl_csv) if rl_csv.exists() else []

    if not greedy_rows:
        print(f'No coalition data found in {greedy_csv}. Run run_coalition.py first.')
        return

    print(f'Loaded {len(greedy_rows)} scenarios from {greedy_csv}')
    if rl_rows:
        print(f'Loaded {len(rl_rows)} RL scenarios from {rl_csv}')

    created = []
    fns = [
        ('gain_contour.png',           lambda: plot_gain_contour(greedy_rows, out_dir)),
        ('operational_table.png',      lambda: plot_operational_table(greedy_rows, out_dir)),
        ('participation_rate_bar.png', lambda: plot_participation_rate_bar(greedy_rows, out_dir)),
        ('ontime_comparison.png',      lambda: plot_ontime_comparison(greedy_rows, out_dir)),
        ('benefit_by_method.png',      lambda: plot_benefit_by_method(greedy_rows, out_dir)),
        ('conviction_bar.png',         lambda: plot_conviction_bar(greedy_rows, out_dir)),
        ('gain_by_scenario.png',       lambda: plot_gain_by_scenario(greedy_rows, out_dir)),
        ('core_rate_heatmap.png',      lambda: plot_core_rate_heatmap(greedy_rows, out_dir)),
        ('factor_sensitivity.png',     lambda: plot_factor_sensitivity(greedy_rows, out_dir)),
        ('gain_distribution.png',      lambda: plot_gain_distribution(greedy_rows, out_dir)),
    ]
    if rl_rows:
        fns.append(('rl_vs_greedy.png', lambda: plot_rl_vs_greedy(greedy_rows, rl_rows, out_dir)))

    for fname, fn in fns:
        try:
            p = fn()
            if p:
                created.append(p)
                print(f'  Saved: {p.name}')
        except Exception as exc:
            print(f'  SKIP {fname}: {exc}')

    print(f'\nTotal: {len(created)} coalition plots saved to {out_dir}')

if __name__ == '__main__':
    main()
