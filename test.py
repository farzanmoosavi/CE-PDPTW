from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

COLOR_UAV = "#1f77b4"
COLOR_ADR = "#2ca02c"
COLOR_LATE = "#d62728"
COLOR_PICKUP = "#ff7f0e"
COLOR_DELIVERY = "#1a1a1a"
COLOR_EPOCH = "#bdbdbd"
COLOR_ARRIVAL = "#7f7f7f"

LANE_HEIGHT = 0.7
EPOCH_LINE_ALPHA = 0.25
LEG_BAR_ALPHA = 0.55

@dataclass
class _LegRender:
    request_id: int
    vehicle_id: int
    mode: str
    leg_type: str
    t_depart: float
    t_arrive: float
    is_late: bool

def _vehicle_mode(vehicle, fleet: Dict[int, object]) -> str:
    v = fleet.get(vehicle)
    if v is None:
        return "uav"
    return getattr(v, "mode", "uav")

def _gather_legs(episode_log: Sequence[dict],
                 fleet: Dict[int, object],
                 request_targets: Dict[int, float]) -> List[_LegRender]:
    out: List[_LegRender] = []
    for entry in episode_log:
        if entry.get("summary"):
            continue
        completed = entry.get("completed", [])
        for item in completed:
            rid = int(item["request_id"])
            vid = int(item["vehicle_id"])
            ltype = item["leg_type"]
            t_arr = float(item["t_arrive"])
            t_dep = t_arr - float(item.get("travel_time", 0.0))
            mode = _vehicle_mode(vid, fleet)
            is_late = False
            if ltype == "delivery":
                t_d = request_targets.get(rid)
                if t_d is not None and t_arr > t_d:
                    is_late = True
            out.append(_LegRender(
                request_id=rid, vehicle_id=vid, mode=mode,
                leg_type=ltype, t_depart=t_dep, t_arrive=t_arr,
                is_late=is_late,
            ))
    return out

def _extract_request_targets(episode_log: Sequence[dict]) -> Dict[int, float]:
    summary = next((e for e in episode_log if e.get("summary")), None)
    if summary is None:
        return {}
    out: Dict[int, float] = {}
    for r in summary.get("request_history", []):
        out[int(r["req_id"])] = float(r["t_delivery_target"])
    return out

def _extract_pickup_targets(episode_log: Sequence[dict]) -> Dict[int, float]:
    summary = next((e for e in episode_log if e.get("summary")), None)
    if summary is None:
        return {}
    return {int(r["req_id"]): float(r["t_pickup_target"])
            for r in summary.get("request_history", [])}

def _extract_arrival_times(episode_log: Sequence[dict],
                           arrival_stream: Optional[Sequence[dict]] = None,
                           ) -> List[float]:
    if arrival_stream is not None:
        return sorted(float(a["t_arrival"]) for a in arrival_stream)
    return []

def _extract_epoch_times(episode_log: Sequence[dict]) -> List[float]:
    return [float(e["t"]) for e in episode_log if not e.get("summary")]

def plot_shift_timeline(
        episode_log: Sequence[dict],
        fleet: Dict[int, object],
        *,
        ax,
        arrival_stream: Optional[Sequence[dict]] = None,
        show_arrivals: bool = True,
        show_epochs: bool = True,
        show_pickup_markers: bool = True,
        show_delivery_markers: bool = True,
        annotate_late: bool = True,
        title: Optional[str] = None,
        shift_minutes: Optional[float] = None,
) -> None:
    delivery_targets = _extract_request_targets(episode_log)
    pickup_targets = _extract_pickup_targets(episode_log)
    legs = _gather_legs(episode_log, fleet, delivery_targets)
    arrival_times = _extract_arrival_times(episode_log, arrival_stream)
    epoch_times = _extract_epoch_times(episode_log)

    vehicle_ids = sorted(fleet.keys())
    n_vehicles = len(vehicle_ids)
    if shift_minutes is None:
        shift_minutes = max(epoch_times or [120.0])

    ordered = sorted(
        vehicle_ids,
        key=lambda vid: (_vehicle_mode(vid, fleet) == "adr", vid),
    )
    y_of_vehicle = {vid: i for i, vid in enumerate(ordered)}

    if show_epochs:
        for t in epoch_times:
            ax.axvline(t, color=COLOR_EPOCH, alpha=EPOCH_LINE_ALPHA,
                       linewidth=0.5, zorder=1)
    if show_arrivals and arrival_times:
        ax.scatter(
            arrival_times,
            [n_vehicles - 0.15] * len(arrival_times),
            marker="v", s=18, color=COLOR_ARRIVAL,
            alpha=0.65, zorder=2,
        )

    for vid in ordered:
        y = y_of_vehicle[vid]
        mode = _vehicle_mode(vid, fleet)
        col = COLOR_UAV if mode == "uav" else COLOR_ADR
        ax.axhspan(y - LANE_HEIGHT / 2, y + LANE_HEIGHT / 2,
                   facecolor=col, alpha=0.05, zorder=0)
        ax.text(-0.5, y, f"V{vid} ({mode.upper()})",
                ha="right", va="center", fontsize=8,
                color=col, fontweight="bold")

    for leg in legs:
        y = y_of_vehicle[leg.vehicle_id]
        mode_col = COLOR_UAV if leg.mode == "uav" else COLOR_ADR
        bar_col = COLOR_LATE if leg.is_late else mode_col
        ax.barh(
            y,
            width=leg.t_arrive - leg.t_depart,
            left=leg.t_depart,
            height=LANE_HEIGHT * 0.55,
            color=bar_col, alpha=LEG_BAR_ALPHA,
            edgecolor=bar_col, linewidth=0.6,
            zorder=3,
        )
        if leg.leg_type == "pickup" and show_pickup_markers:
            ax.scatter(leg.t_arrive, y, marker="o", s=42,
                       facecolor=COLOR_PICKUP, edgecolor="black",
                       linewidth=0.5, zorder=4)
        elif leg.leg_type == "delivery" and show_delivery_markers:
            edge = COLOR_LATE if leg.is_late else COLOR_DELIVERY
            face = COLOR_LATE if leg.is_late else "white"
            ax.scatter(leg.t_arrive, y, marker="s", s=44,
                       facecolor=face, edgecolor=edge,
                       linewidth=1.0, zorder=4)
            if annotate_late and leg.is_late:
                t_target = delivery_targets.get(leg.request_id)
                if t_target is not None:
                    ax.plot([t_target, leg.t_arrive], [y, y],
                            color=COLOR_LATE, linewidth=1.5,
                            linestyle=":", zorder=3.5)
                    ax.annotate(
                        f"+{leg.t_arrive - t_target:.0f}m",
                        xy=(leg.t_arrive, y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        color=COLOR_LATE,
                        fontsize=7,
                        fontweight="bold",
                    )

    ax.set_xlim(-1, shift_minutes + 1)
    ax.set_ylim(-0.6, n_vehicles - 0.4)
    ax.set_xlabel("Time within shift (min)")
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    _add_legend(ax)

def _add_legend(ax) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=COLOR_UAV, alpha=LEG_BAR_ALPHA, label="UAV leg"),
        Patch(facecolor=COLOR_ADR, alpha=LEG_BAR_ALPHA, label="ADR leg"),
        Patch(facecolor=COLOR_LATE, alpha=LEG_BAR_ALPHA, label="Late delivery"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLOR_PICKUP, markeredgecolor="black",
               markersize=7, label="Pickup completion"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor="white", markeredgecolor=COLOR_DELIVERY,
               markersize=7, markeredgewidth=1.0, label="Delivery completion"),
        Line2D([0], [0], marker="v", color="w",
               markerfacecolor=COLOR_ARRIVAL, markersize=6,
               label="Request arrival"),
        Line2D([0], [0], color=COLOR_EPOCH, linewidth=0.8,
               linestyle="-", label="Decision epoch"),
    ]
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=True)

def plot_shift_comparison(
        logs_by_method: Dict[str, Sequence[dict]],
        fleet: Dict[int, object],
        *,
        arrival_stream: Optional[Sequence[dict]] = None,
        figsize: Tuple[float, float] = (14, 9),
):
    import matplotlib.pyplot as plt
    methods = list(logs_by_method.keys())
    fig, axes = plt.subplots(len(methods), 1, figsize=figsize, sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        plot_shift_timeline(
            logs_by_method[method], fleet, ax=ax,
            arrival_stream=arrival_stream,
            title=method,
        )
    fig.tight_layout()
    return fig, axes

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    class _DummyVeh:
        def __init__(self, mode): self.mode = mode

    fleet = {0: _DummyVeh("uav"), 1: _DummyVeh("uav"),
             2: _DummyVeh("adr"), 3: _DummyVeh("adr")}

    log = [
        {"t": 5.0, "completed": [
            {"request_id": 0, "vehicle_id": 0, "leg_type": "pickup",
             "t_arrive": 4.2, "travel_time": 1.5},
        ]},
        {"t": 10.0, "completed": [
            {"request_id": 0, "vehicle_id": 0, "leg_type": "delivery",
             "t_arrive": 9.6, "travel_time": 4.2},
            {"request_id": 1, "vehicle_id": 2, "leg_type": "pickup",
             "t_arrive": 9.1, "travel_time": 2.0},
        ]},
        {"t": 15.0, "completed": [
            {"request_id": 1, "vehicle_id": 2, "leg_type": "delivery",
             "t_arrive": 14.8, "travel_time": 5.1},
        ]},
        {"t": 20.0, "completed": []},
        {"summary": True,
         "request_history": [
             {"req_id": 0, "t_pickup_target": 4.0, "t_delivery_target": 10.0},
             {"req_id": 1, "t_pickup_target": 9.0, "t_delivery_target": 12.0},
         ]},
    ]

    arrivals = [
        {"t_arrival": 1.0}, {"t_arrival": 5.5}, {"t_arrival": 8.0},
    ]

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_shift_timeline(log, fleet, ax=ax, arrival_stream=arrivals,
                        title="Smoke test")
    fig.savefig("shift_timeline_smoke.png",
                bbox_inches="tight", dpi=120)
    print("Wrote /tmp/shift_timeline_smoke.png")



  # Phase 1 — close the training-inference gap (highest ROI):
  #   [x] full episode training loop (simulate shifts, not static batches)
  #   [x] GAE with simple value MLP on concat(vehicle_states)
  #   [x] hard trigger: re-encode on any new arrival
  #   [x] t_current/120 as 14th node feature (trivial, free peak signal)
  #
  # Phase 2 — add live vehicle state:
  #   [ ] VehicleGRU (3-input, d=32)
  #   [ ] decoder query = HetGAT_last_node_emb + h_v_t  (concat, project)
  #
  # Phase 3 — global context (if phase 2 doesn't saturate):
  #   [ ] GlobalGRU feeding c_t to decoder cross-attention
  #   [ ] ValueNet on c_t replacing simple value MLP
  #
  # Phase 4 — learned trigger (research contribution, not engineering):
  #   [ ] Trigger MLP on c_t with penalty-augmented reward
