from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

def _draw_basemap(ax, networks, alpha_road=0.25):
    try:
        import osmnx as ox
        from shapely.geometry import Polygon
        from mississauga_instance import (
            MISSISSAUGA_POLYGON_LONLAT,
            ADR_NETWORK_FILTER,
        )
        poly = Polygon(MISSISSAUGA_POLYGON_LONLAT)
        G = ox.graph_from_polygon(poly, simplify=True, retain_all=False,
                                   network_type="all",
                                   custom_filter=ADR_NETWORK_FILTER)
        ox.plot_graph(G, ax=ax, edge_color="#bdbdbd", node_size=0,
                      show=False, close=False, edge_linewidth=0.5,
                      bgcolor="#ffffff")
    except Exception as e:
        ax.scatter(networks.coords_km[:, 0], networks.coords_km[:, 1],
                   s=1, color="#dddddd", alpha=alpha_road, zorder=0)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

def _draw_pearson_no_fly(ax, networks):
    from mississauga_instance import PEARSON_NO_FLY_RADIUS_KM
    cx, cy = float(networks.pearson_xy_km[0]), float(networks.pearson_xy_km[1])
    circle = _matplotlib_circle((cx, cy), PEARSON_NO_FLY_RADIUS_KM,
                                  facecolor="red", edgecolor="red",
                                  alpha=0.08, linewidth=1.0, linestyle="--",
                                  zorder=1, label="Pearson no-fly")
    ax.add_patch(circle)

def _matplotlib_circle(xy, radius, **kwargs):
    from matplotlib.patches import Circle
    return Circle(xy, radius, **kwargs)

def _extract_delivery_assignments(inst, route_plan,
                                    n_uav: int) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for vid, legs in route_plan.items():
        mode = "uav" if vid < n_uav else "adr"
        for leg in legs:
            if getattr(leg, "leg_type", None) == "delivery":
                out[int(leg.request_id)] = mode
    return out

def _instance_geometry(inst) -> Dict[str, np.ndarray]:
    if "coords_km" not in inst:
        raise KeyError(
            "Instance dict does not contain 'coords_km'. "
            "Pass an instance produced by mississauga_instance.create_mississauga_instance."
        )
    c = inst["coords_km"]
    coords = c.cpu().numpy() if hasattr(c, "cpu") else np.asarray(c)
    n_depots = int(inst["n_depots"])
    n_req = int(inst["n_req"])
    return {
        "coords_km": coords,
        "depots_xy": coords[:n_depots],
        "depots_uav_xy": coords[: int(inst["n_depots_uav"])],
        "depots_adr_xy": coords[int(inst["n_depots_uav"]): n_depots],
        "pickup_xy": coords[n_depots:n_depots + n_req],
        "delivery_xy": coords[n_depots + n_req:],
        "n_depots": n_depots,
        "n_req": n_req,
    }

def plot_instance_mode_coverage(
    inst,
    route_plan,
    *,
    networks,
    ax,
    n_uav: int = 3,
    title: Optional[str] = None,
    show_no_fly: bool = True,
    legend: bool = True,
) -> None:
    _draw_basemap(ax, networks)
    if show_no_fly:
        _draw_pearson_no_fly(ax, networks)

    g = _instance_geometry(inst)
    assignments = _extract_delivery_assignments(inst, route_plan, n_uav=n_uav)

    uav_xy, adr_xy, undel_xy = [], [], []
    for req_id in range(g["n_req"]):
        xy = g["delivery_xy"][req_id]
        mode = assignments.get(req_id, "undelivered")
        if mode == "uav":
            uav_xy.append(xy)
        elif mode == "adr":
            adr_xy.append(xy)
        else:
            undel_xy.append(xy)

    if uav_xy:
        uav_xy = np.array(uav_xy)
        ax.scatter(uav_xy[:, 0], uav_xy[:, 1], s=60,
                   color="#1f77b4", marker="o", edgecolor="black",
                   linewidth=0.6, label="Delivery — UAV", zorder=4)
    if adr_xy:
        adr_xy = np.array(adr_xy)
        ax.scatter(adr_xy[:, 0], adr_xy[:, 1], s=60,
                   color="#2ca02c", marker="s", edgecolor="black",
                   linewidth=0.6, label="Delivery — ADR", zorder=4)
    if undel_xy:
        undel_xy = np.array(undel_xy)
        ax.scatter(undel_xy[:, 0], undel_xy[:, 1], s=70,
                   facecolor="none", edgecolor="#d62728", marker="X",
                   linewidth=1.4, label="Undelivered", zorder=4)

    ax.scatter(g["pickup_xy"][:, 0], g["pickup_xy"][:, 1],
               s=35, color="#ff7f0e", marker="*", edgecolor="black",
               linewidth=0.4, label="Pickup (restaurant)", zorder=3)

    if len(g["depots_uav_xy"]):
        ax.scatter(g["depots_uav_xy"][:, 0], g["depots_uav_xy"][:, 1],
                   s=140, color="#1f77b4", marker="^", edgecolor="black",
                   linewidth=0.8, label="UAV depot", zorder=5)
    if len(g["depots_adr_xy"]):
        ax.scatter(g["depots_adr_xy"][:, 0], g["depots_adr_xy"][:, 1],
                   s=140, color="#2ca02c", marker="P", edgecolor="black",
                   linewidth=0.8, label="ADR depot", zorder=5)

    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")

def plot_mode_assignment_aggregate(
    instances: Sequence,
    route_plans: Sequence,
    *,
    networks,
    ax,
    n_uav: int = 3,
    title: Optional[str] = None,
    grid_km: float = 0.5,
    show_no_fly: bool = True,
) -> None:
    assert len(instances) == len(route_plans), \
        "instances and route_plans must align."

    _draw_basemap(ax, networks)
    if show_no_fly:
        _draw_pearson_no_fly(ax, networks)

    xmin, ymin, xmax, ymax = networks.bbox_xy_km
    nx_bins = max(1, int(np.ceil((xmax - xmin) / grid_km)))
    ny_bins = max(1, int(np.ceil((ymax - ymin) / grid_km)))
    counts_uav = np.zeros((ny_bins, nx_bins), dtype=np.int32)
    counts_adr = np.zeros((ny_bins, nx_bins), dtype=np.int32)

    for inst, plan in zip(instances, route_plans):
        g = _instance_geometry(inst)
        assignments = _extract_delivery_assignments(inst, plan, n_uav=n_uav)
        for req_id in range(g["n_req"]):
            mode = assignments.get(req_id)
            if mode not in ("uav", "adr"):
                continue
            x, y = g["delivery_xy"][req_id]
            ix = int(np.clip((x - xmin) / grid_km, 0, nx_bins - 1))
            iy = int(np.clip((y - ymin) / grid_km, 0, ny_bins - 1))
            if mode == "uav":
                counts_uav[iy, ix] += 1
            else:
                counts_adr[iy, ix] += 1

    total = counts_uav + counts_adr
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_uav = np.where(total > 0, counts_uav / np.maximum(total, 1), np.nan)

    cmap = _build_uav_adr_colormap()
    extent = (xmin, xmin + nx_bins * grid_km,
              ymin, ymin + ny_bins * grid_km)
    img = ax.imshow(frac_uav, origin="lower", extent=extent,
                     cmap=cmap, vmin=0.0, vmax=1.0,
                     alpha=0.65, interpolation="nearest", zorder=2)

    cbar = ax.figure.colorbar(img, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("UAV-served fraction")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["ADR", "0.25", "0.5", "0.75", "UAV"])

    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")

def _build_uav_adr_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "uav_adr",
        [(0.0, "#2ca02c"), (0.5, "#f0f0f0"), (1.0, "#1f77b4")],
        N=256,
    )

def episode_log_to_route_plan(episode_log: List[Dict]) -> Dict[int, list]:
    from types import SimpleNamespace

    route_plan: Dict[int, list] = defaultdict(list)
    for entry in episode_log:
        if entry.get("summary"):
            continue
        for leg_dict in entry.get("completed", []):
            vid = int(leg_dict["vehicle_id"])
            proxy = SimpleNamespace(
                request_id=int(leg_dict["request_id"]),
                leg_type=str(leg_dict["leg_type"]),
            )
            route_plan[vid].append(proxy)
    return dict(route_plan)

def summarise_mode_assignment(instances, route_plans, *, n_uav: int) -> Dict:
    n_uav_served = 0
    n_adr_served = 0
    n_undelivered = 0
    payload_uav = []
    payload_adr = []
    for inst, plan in zip(instances, route_plans):
        g = _instance_geometry(inst)
        demand = inst["demand"].cpu().numpy().flatten()
        n_depots = g["n_depots"]
        assignments = _extract_delivery_assignments(inst, plan, n_uav=n_uav)
        for r in range(g["n_req"]):
            q = float(demand[n_depots + r])
            mode = assignments.get(r, "undelivered")
            if mode == "uav":
                n_uav_served += 1
                payload_uav.append(q)
            elif mode == "adr":
                n_adr_served += 1
                payload_adr.append(q)
            else:
                n_undelivered += 1
    return {
        "n_uav_served": n_uav_served,
        "n_adr_served": n_adr_served,
        "n_undelivered": n_undelivered,
        "mean_payload_uav": float(np.mean(payload_uav)) if payload_uav else 0.0,
        "mean_payload_adr": float(np.mean(payload_adr)) if payload_adr else 0.0,
        "uav_fraction": n_uav_served / max(1, n_uav_served + n_adr_served),
    }
