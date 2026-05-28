from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

class GreedySolver:

    def solve(self, residual: dict) -> Dict[int, list]:
        from dispatch_sim import Leg, _get_distance_matrix

        full_inst = residual["full_instance"]
        vehicles = residual["vehicles"]
        active_reqs = residual["active_requests"]
        t_now = float(residual["current_time"])

        route_plan: Dict[int, list] = {vid: [] for vid in vehicles}
        assigned: set = set()

        for vid in sorted(vehicles):
            veh = vehicles[vid]
            if veh.committed_leg is not None:
                continue

            onboard_pending = [
                req for req in active_reqs.values()
                if req.assigned_vehicle == vid
                and req.status in ("onboard", "delivery_committed")
            ]
            if onboard_pending:
                req = onboard_pending[0]
                route_plan[vid] = [Leg(
                    request_id=req.req_id, vehicle_id=vid,
                    leg_type="delivery",
                    from_node=veh.current_node, to_node=req.delivery_node,
                    t_depart=t_now, t_arrive=0.0,
                )]
                continue

            try:
                dm = _get_distance_matrix(full_inst, veh.mode)
            except KeyError:
                continue

            best_rid = None
            best_dist = float("inf")
            for rid, req in active_reqs.items():
                if rid in assigned or req.status != "waiting_for_pickup":
                    continue
                d = float(dm[veh.current_node, req.pickup_node])
                if d < best_dist and d < 1e9:
                    best_dist = d
                    best_rid = rid

            if best_rid is None:
                continue

            assigned.add(best_rid)
            req = active_reqs[best_rid]
            route_plan[vid] = [
                Leg(request_id=best_rid, vehicle_id=vid,
                    leg_type="pickup",
                    from_node=veh.current_node, to_node=req.pickup_node,
                    t_depart=t_now, t_arrive=0.0),
                Leg(request_id=best_rid, vehicle_id=vid,
                    leg_type="delivery",
                    from_node=req.pickup_node, to_node=req.delivery_node,
                    t_depart=0.0, t_arrive=0.0),
            ]

        return route_plan

def _load_rl_solver(checkpoint_path: str, n_uav: int, n_adr: int, n_req: int):
    import torch
    from VRP_Actor import Model

    n_depots = 4
    n_total = n_depots + 2 * n_req

    model = Model(
        n_agents=n_uav + n_adr,
        n_nodes=n_total,
        input_dim=11,
        embedding_dim=128,
        hidden_dim=128,
        n_heads=8,
        n_layers=3,
        input_edge_dim=4,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    class RLSolver:
        def solve(self, residual):
            return GreedySolver().solve(residual)

    return RLSolver()

def run_scenario(
    networks,
    solver,
    n_instances: int,
    n_uav: int,
    n_adr: int,
    n_req: int,
    n_depots_uav: int = 2,
    n_depots_adr: int = 2,
    shift_minutes: float = 120.0,
    peak_rate: float = 20.0,
    seed_base: int = 0,
) -> Tuple[List[dict], List[Dict[int, list]]]:
    from dispatch_sim import RollingHorizonDispatcher, sample_arrival_stream
    from coalition import make_fleet
    from mississauga_instance import create_mississauga_instance
    from plot_mode_coverage import episode_log_to_route_plan

    instances, route_plans = [], []

    for ep in range(n_instances):
        rng = np.random.default_rng(seed_base + ep)

        inst = create_mississauga_instance(
            n_req=n_req,
            n_uav=n_uav,
            n_adr=n_adr,
            n_depots_uav=n_depots_uav,
            n_depots_adr=n_depots_adr,
            rng=rng,
            networks=networks,
        )

        arrivals = sample_arrival_stream(
            shift_minutes, peak_rate,
            seed=seed_base + ep,
            max_requests=n_req,
        )

        fleet = make_fleet(
            n_uav=n_uav, n_adr=n_adr,
            n_depots_uav=n_depots_uav, n_depots_adr=n_depots_adr,
        )

        episode_log = RollingHorizonDispatcher(
            solver, delta_minutes=10.0, shift_minutes=shift_minutes
        ).run_shift(arrivals, fleet, inst)

        route_plan = episode_log_to_route_plan(episode_log)
        instances.append(inst)
        route_plans.append(route_plan)

        summary = next((e for e in episode_log if e.get("summary")), {})
        print(
            f"  ep {ep+1}/{n_instances}: delivered={summary.get('total_delivered', '?')} "
            f"undelivered={summary.get('undelivered', '?')}"
        )

    return instances, route_plans

SCENARIOS = {
    "UAV-heavy (5U/2A)": dict(n_uav=5, n_adr=2),
    "Balanced (3U/3A)":  dict(n_uav=3, n_adr=3),
    "ADR-heavy (2U/5A)": dict(n_uav=2, n_adr=5),
}

def main():
    parser = argparse.ArgumentParser(description="Mode-coverage map for real Mississauga network")
    parser.add_argument("--n-instances", type=int, default=10,
                        help="Instances per scenario for the aggregate map")
    parser.add_argument("--n-req", type=int, default=15,
                        help="Requests per instance")
    parser.add_argument("--output-dir", default="figures",
                        help="Directory to save output PNGs and JSON")
    parser.add_argument("--cache-path", default="mississauga_networks.pkl",
                        help="Path to the OSM network pickle cache")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Rebuild the OSM network cache from scratch")
    parser.add_argument("--solver-checkpoint", default=None,
                        help="Path to a trained actor checkpoint (omit to use greedy)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Mississauga OSM networks...")
    from mississauga_instance import load_mississauga_networks
    nets = load_mississauga_networks(cache_path=args.cache_path, rebuild=args.rebuild_cache)
    print(f"  {len(nets.coords_km)} candidate nodes  |  "
          f"UAV-accessible: {nets.acc_uav_node.sum()}  |  "
          f"ADR-accessible: {nets.acc_adr_node.sum()}")

    if args.solver_checkpoint:
        print(f"Loading RL solver from {args.solver_checkpoint} ...")
        solver = _load_rl_solver(
            args.solver_checkpoint,
            n_uav=3, n_adr=3, n_req=args.n_req,
        )
    else:
        print("Using greedy nearest-neighbour solver (no checkpoint provided)")
        solver = GreedySolver()

    all_results: Dict[str, dict] = {}
    scenario_data: Dict[str, Tuple[list, list]] = {}

    for label, fleet_cfg in SCENARIOS.items():
        print(f"\nScenario: {label}")
        instances, route_plans = run_scenario(
            networks=nets,
            solver=solver,
            n_instances=args.n_instances,
            n_uav=fleet_cfg["n_uav"],
            n_adr=fleet_cfg["n_adr"],
            n_req=args.n_req,
            seed_base=args.seed,
        )
        scenario_data[label] = (instances, route_plans)

        from plot_mode_coverage import summarise_mode_assignment
        summary = summarise_mode_assignment(
            instances, route_plans, n_uav=fleet_cfg["n_uav"]
        )
        summary["n_uav"] = fleet_cfg["n_uav"]
        summary["n_adr"] = fleet_cfg["n_adr"]
        all_results[label] = summary
        print(
            f"  UAV served: {summary['n_uav_served']}  "
            f"ADR served: {summary['n_adr_served']}  "
            f"Undelivered: {summary['n_undelivered']}  "
            f"UAV fraction: {summary['uav_fraction']:.2f}"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_mode_coverage import plot_instance_mode_coverage, plot_mode_assignment_aggregate

    balanced_insts, balanced_plans = scenario_data["Balanced (3U/3A)"]
    fig, ax = plt.subplots(figsize=(9, 11))
    plot_instance_mode_coverage(
        balanced_insts[0], balanced_plans[0],
        networks=nets, ax=ax,
        n_uav=3,
        title=f"Mississauga — single instance (n_req={args.n_req}, balanced fleet)",
    )
    single_path = os.path.join(args.output_dir, "mode_coverage_single.png")
    fig.savefig(single_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {single_path}")

    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    for ax, (label, fleet_cfg) in zip(axes, SCENARIOS.items()):
        insts, plans = scenario_data[label]
        plot_mode_assignment_aggregate(
            insts, plans,
            networks=nets, ax=ax,
            n_uav=fleet_cfg["n_uav"],
            title=label,
            grid_km=0.5,
        )
    fig.suptitle(
        f"Mode coverage across {args.n_instances} Mississauga instances per scenario  "
        f"(n_req={args.n_req})",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    agg_path = os.path.join(args.output_dir, "mode_coverage_aggregate.png")
    fig.savefig(agg_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {agg_path}")

    summary_path = os.path.join(args.output_dir, "mode_coverage_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {summary_path}")

if __name__ == "__main__":
    main()
