from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt

PLOT_METRICS = [
    ("total_cost", "Total cost"),
    ("delivered_rate", "Delivered rate"),
    ("hard_constraint_violation_rate", "Hard constraint violation rate"),
    ("soft_time_window_violation_rate", "Soft time-window violation rate"),
    ("wall_time_s", "Wall-clock runtime (s)"),
    ("solver_time_s_total", "Solver planning time (s)"),
    ("arc_constraint_violation_rate", "Arc violation rate"),
    ("capacity_violation_rate", "Capacity violation rate"),
    ("depot_sharing_violation_rate", "Depot-sharing violation rate"),
    ("battery_threshold_violation_epoch_rate", "Battery-threshold violation rate"),
]

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))

def to_float(value: object) -> float | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number

def group_values(rows: Sequence[Dict[str, str]], metric: str) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = {}

    for row in rows:
        if str(row.get("available", "")).lower() not in {"true", "1", "yes"}:
            continue

        baseline = str(row.get("baseline", "")).strip()
        if not baseline:
            continue

        value = to_float(row.get(metric))
        if value is None:
            continue

        grouped.setdefault(baseline, []).append(value)

    return grouped

def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))

def make_bar_plot(
    grouped: Dict[str, List[float]],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    show_error_bars: bool = True,
) -> None:
    if not grouped:
        return

    labels = sorted(grouped)
    averages = [mean(grouped[label]) for label in labels]
    errors = [std(grouped[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(len(labels)))

    if show_error_bars and any(error > 0 for error in errors):
        ax.bar(positions, averages, yerr=errors, capsize=4)
    else:
        ax.bar(positions, averages)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def make_scatter_by_instance(
    rows: Sequence[Dict[str, str]],
    metric: str,
    *,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    clean_rows = [
        row for row in rows
        if str(row.get("available", "")).lower() in {"true", "1", "yes"}
        and to_float(row.get(metric)) is not None
        and to_float(row.get("instance_id")) is not None
    ]

    if not clean_rows:
        return

    baselines = sorted({str(row.get("baseline", "")).strip() for row in clean_rows if row.get("baseline")})
    fig, ax = plt.subplots(figsize=(8, 5))

    for baseline in baselines:
        xs = []
        ys = []
        for row in clean_rows:
            if str(row.get("baseline", "")).strip() != baseline:
                continue
            x = to_float(row.get("instance_id"))
            y = to_float(row.get(metric))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if xs:
            ax.scatter(xs, ys, label=baseline)

    ax.set_title(title)
    ax.set_xlabel("Instance ID")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def make_component_cost_plot(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    components = [
        ("operating_cost", "Operating"),
        ("penalty_cost", "Penalty"),
        ("battery_penalty_cost", "Battery penalty"),
    ]

    baselines = sorted({
        str(row.get("baseline", "")).strip()
        for row in rows
        if str(row.get("available", "")).lower() in {"true", "1", "yes"} and row.get("baseline")
    })

    if not baselines:
        return

    component_means: Dict[str, List[float]] = {}
    for metric, _ in components:
        grouped = group_values(rows, metric)
        component_means[metric] = [mean(grouped.get(baseline, [])) for baseline in baselines]

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = list(range(len(baselines)))
    bottoms = [0.0 for _ in baselines]

    for metric, label in components:
        values = component_means[metric]
        ax.bar(positions, values, bottom=bottoms, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_title("Mean cost components by baseline")
    ax.set_ylabel("Cost")
    ax.set_xticks(positions)
    ax.set_xticklabels(baselines, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def make_violation_summary_plot(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    metrics = [
        ("hard_constraint_violation_rate", "Hard"),
        ("soft_time_window_violation_rate", "Soft TW"),
        ("arc_constraint_violation_rate", "Arc"),
        ("capacity_violation_rate", "Capacity"),
        ("depot_sharing_violation_rate", "Depot"),
        ("battery_threshold_violation_epoch_rate", "Battery"),
    ]

    baselines = sorted({
        str(row.get("baseline", "")).strip()
        for row in rows
        if str(row.get("available", "")).lower() in {"true", "1", "yes"} and row.get("baseline")
    })

    if not baselines:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x_base = list(range(len(baselines)))
    width = 0.12

    for metric_index, (metric, label) in enumerate(metrics):
        grouped = group_values(rows, metric)
        values = [mean(grouped.get(baseline, [])) for baseline in baselines]
        offsets = [x + (metric_index - (len(metrics) - 1) / 2) * width for x in x_base]
        ax.bar(offsets, values, width=width, label=label)

    ax.set_title("Mean violation rates by baseline")
    ax.set_ylabel("Violation rate")
    ax.set_xticks(x_base)
    ax.set_xticklabels(baselines, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def write_plot_index(output_dir: Path, plot_paths: Iterable[Path]) -> None:
    lines = ["# Rolling Baseline Plots", ""]
    for path in sorted(plot_paths):
        rel = path.relative_to(output_dir)
        lines.append(f"- `{rel}`")
    lines.append("")
    (output_dir / "plots" / "README.md").write_text("\n".join(lines), encoding="utf-8")

def plot_all_metrics(results_dir: str | Path) -> List[Path]:
    results_dir = Path(results_dir)
    per_instance_path = results_dir / "per_instance_results.csv"
    rows = read_csv_rows(per_instance_path)

    plots_dir = results_dir / "plots"
    created: List[Path] = []

    for metric, title in PLOT_METRICS:
        grouped = group_values(rows, metric)

        output_path = plots_dir / f"{metric}_bar.png"
        make_bar_plot(
            grouped,
            title=f"Mean {title} by baseline",
            ylabel=title,
            output_path=output_path,
        )
        if output_path.exists():
            created.append(output_path)

        scatter_path = plots_dir / f"{metric}_by_instance.png"
        make_scatter_by_instance(
            rows,
            metric,
            title=f"{title} by instance",
            ylabel=title,
            output_path=scatter_path,
        )
        if scatter_path.exists():
            created.append(scatter_path)

    component_path = plots_dir / "cost_components_stacked.png"
    make_component_cost_plot(rows, component_path)
    if component_path.exists():
        created.append(component_path)

    violation_path = plots_dir / "violation_rates_grouped.png"
    make_violation_summary_plot(rows, violation_path)
    if violation_path.exists():
        created.append(violation_path)

    write_plot_index(results_dir, created)
    return created

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="rolling_benchmark_results")
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    created = plot_all_metrics(Path(args.results_dir))
    print(f"Created {len(created)} plots in {Path(args.results_dir) / 'plots'}")
    for path in created:
        print(path)

if __name__ == "__main__":
    main()
