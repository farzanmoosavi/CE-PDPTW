import sys
import argparse
from typing import Dict, List, Optional, Tuple

def build_solver(model_path: str, arch: str = 'hetgat'):
    import torch
    from VRP_Actor import Model
    from dispatch_sim import Leg, Request

    if not model_path:
        raise ValueError(
            'A trained model checkpoint is required. Pass --model path/to/actor.pt'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _raw = torch.load(model_path, map_location=device, weights_only=False)
    _state = _raw['model_state_dict'] if isinstance(_raw, dict) and 'model_state_dict' in _raw else _raw
    _input_node_dim = 11
    for _k, _v in _state.items():
        if _k.endswith('encoder.W_depot.weight'):
            _input_node_dim = int(_v.shape[1])
            break
    _dynamic_features = (_input_node_dim == 13)
    edge_dim = 1 if arch == 'simplegat' else 4
    actor = Model(
        input_node_dim=_input_node_dim,
        hidden_node_dim=128,
        input_edge_dim=edge_dim,
        hidden_edge_dim=16,
        conv_layers=3,
        arch=arch,
    ).to(device)
    if isinstance(_raw, dict) and 'model_state_dict' in _raw:
        actor.load_state_dict(_raw['model_state_dict'])
        print(f'Loaded checkpoint from {model_path} (epoch {_raw.get("epoch", "?")}, '
              f'input_node_dim={_input_node_dim})')
    else:
        actor.load_state_dict(_raw)
        print(f'Loaded model weights from {model_path} (input_node_dim={_input_node_dim})')
    actor.eval()

    class RLSolver:
        def __init__(self, actor_model, torch_device, dynamic_features=False):
            self.actor = actor_model
            self.device = torch_device
            self.dynamic_features = dynamic_features

        def _batchify_instance(self, instance: dict) -> dict:
            batched = {}
            for key, value in instance.items():
                if torch.is_tensor(value):
                    batched[key] = value.unsqueeze(0).to(self.device)
                else:
                    batched[key] = value
            return batched

        def _num_agents(self, vehicles: Dict[int, object]) -> Tuple[int, int]:
            n_uav = sum(1 for v in vehicles.values() if v.mode == 'uav')
            n_adr = sum(1 for v in vehicles.values() if v.mode == 'adr')
            return n_uav, n_adr

        def _distance_matrix(self, full_instance: dict, mode: str):
            key = 'edge_attr_d' if mode == 'uav' else 'edge_attr_r'
            raw = full_instance[key]
            n_total = full_instance['x'].shape[0]
            return raw.view(n_total, n_total).detach().cpu()

        def _speed(self, mode: str) -> float:
            from vrpUpdate import V_UAV_MAX, V_ADR_MAX
            return V_UAV_MAX if mode == 'uav' else V_ADR_MAX

        def _travel_time(self, full_instance: dict, mode: str, from_node: int, to_node: int) -> float:
            dm = self._distance_matrix(full_instance, mode)
            d = float(dm[from_node, to_node].item())
            if d >= 1e9:
                return float('inf')
            return d / max(self._speed(mode), 1e-9)

        def _mask_unrevealed_nodes(self, batch: dict, active_requests, full_instance) -> dict:
            import torch
            n_depots = int(full_instance['n_depots'])
            n_total = batch['x'].shape[1]

            revealed_nodes = set(range(n_depots))
            for req in active_requests.values():
                revealed_nodes.add(int(req.pickup_node))
                revealed_nodes.add(int(req.delivery_node))

            unrevealed = [i for i in range(n_depots, n_total) if i not in revealed_nodes]
            if not unrevealed:
                return batch

            batch = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            device = batch['x'].device
            ut = torch.tensor(unrevealed, dtype=torch.long, device=device)

            batch['x'][:, ut, :] = 0.0
            if 'demand' in batch and isinstance(batch['demand'], torch.Tensor):
                batch['demand'][:, ut] = 0.0
            if 'time_window' in batch and isinstance(batch['time_window'], torch.Tensor):
                batch['time_window'][:, ut] = 0.0

            n_sq = n_total * n_total
            for mask_key in ('mask_adjacency_uav', 'mask_adjacency_adr'):
                m = batch.get(mask_key)
                if not isinstance(m, torch.Tensor) or m.dim() != 3 or m.shape[1] != n_sq:
                    continue
                ch = m.shape[-1]
                m2d = m.view(m.shape[0], n_total, n_total, ch)
                m2d[:, ut, :, :] = 0.0
                m2d[:, :, ut, :] = 0.0

            for edge_key in ('edge_attr_uav', 'edge_attr_adr',
                             'edge_attr_d', 'edge_attr_r'):
                e = batch.get(edge_key)
                if not isinstance(e, torch.Tensor) or e.dim() != 3 or e.shape[1] != n_sq:
                    continue
                ch = e.shape[-1]
                e2d = e.view(e.shape[0], n_total, n_total, ch)
                e2d[:, ut, :, :] = 0.0
                e2d[:, :, ut, :] = 0.0

            return batch

        def _node_role(self, node: int, n_depots: int, n_req: int) -> Optional[str]:
            if n_depots <= node < n_depots + n_req:
                return 'pickup'
            if n_depots + n_req <= node < n_depots + 2 * n_req:
                return 'delivery'
            return None

        def _match_request(
            self,
            active_requests: Dict[int, Request],
            node: int,
            leg_type: str,
            vehicle_id: int,
            seen_pickups: set,
            onboard_reqs: set,
        ) -> Optional[Request]:
            matches: List[Request] = []
            for req in active_requests.values():
                target_node = req.pickup_node if leg_type == 'pickup' else req.delivery_node
                if int(target_node) != int(node):
                    continue
                if leg_type == 'pickup':
                    if req.status == 'waiting_for_pickup' and (
                        req.assigned_vehicle is None or req.assigned_vehicle == vehicle_id
                    ):
                        matches.append(req)
                else:
                    if req.status in ('onboard', 'delivery_committed') and req.assigned_vehicle == vehicle_id:
                        matches.append(req)
                    elif req.status == 'pickup_committed' and req.assigned_vehicle == vehicle_id and req.req_id in seen_pickups:
                        matches.append(req)
                    elif req.req_id in onboard_reqs:
                        matches.append(req)
                    elif req.req_id in seen_pickups:
                        matches.append(req)
            if not matches:
                return None
            matches.sort(key=lambda r: (r.t_delivery, r.req_id))
            return matches[0]

        def _decode_route_plan(self, residual: dict, decoded_tour) -> Dict[int, List['Leg']]:
            from dispatch_sim import Leg

            vehicles = residual['vehicles']
            active_requests = residual['active_requests']
            full_instance = residual['full_instance']
            current_time = residual['current_time']

            n_depots = int(full_instance['n_depots'])
            n_req = int(full_instance['n_req'])

            route_plan: Dict[int, List[Leg]] = {vid: [] for vid in vehicles.keys()}
            ordered_vids = sorted(vehicles.keys())

            for local_agent_idx, vid in enumerate(ordered_vids):
                veh = vehicles[vid]
                seq = decoded_tour[local_agent_idx].tolist()

                prev_node = int(veh.current_node)
                depart_time = float(max(current_time, veh.current_time))
                seen_pickups = set()

                for node in seq:
                    node = int(node)
                    if node < n_depots or node == prev_node:
                        continue

                    leg_type = self._node_role(node, n_depots, n_req)
                    if leg_type is None:
                        continue

                    req = self._match_request(
                        active_requests=active_requests,
                        node=node,
                        leg_type=leg_type,
                        vehicle_id=vid,
                        seen_pickups=seen_pickups,
                        onboard_reqs=veh.onboard_requests,
                    )
                    if req is None:
                        continue

                    to_node = int(req.pickup_node if leg_type == 'pickup' else req.delivery_node)
                    tt = self._travel_time(full_instance, veh.mode, prev_node, to_node)
                    if tt == float('inf'):
                        continue

                    leg = Leg(
                        request_id=req.req_id,
                        vehicle_id=vid,
                        leg_type=leg_type,
                        from_node=prev_node,
                        to_node=to_node,
                        t_depart=depart_time,
                        t_arrive=depart_time + tt,
                    )
                    route_plan[vid].append(leg)

                    depart_time += tt
                    prev_node = to_node
                    if leg_type == 'pickup':
                        seen_pickups.add(req.req_id)

            return route_plan

        def _build_initial_visited(self, batch, active_requests, full_instance) -> 'torch.Tensor':
            import torch
            n_total = batch['x'].shape[1]
            n_depots = int(full_instance['n_depots'])
            visited = torch.zeros(1, n_total, device=batch['x'].device)

            # Pre-mark ALL customer nodes as visited so the decoder's termination
            # condition fires correctly.  Unrevealed nodes are never reachable
            # (adjacency zeroed by _mask_unrevealed_nodes) and must not keep the
            # while-loop alive past the point where all active requests are routed.
            visited[0, n_depots:] = 1.0

            for req in active_requests.values():
                p = int(req.pickup_node)
                d = int(req.delivery_node)
                status = req.status
                if status == 'waiting_for_pickup':
                    visited[0, p] = 0.0  # decoder must route pickup and delivery
                    visited[0, d] = 0.0
                elif status in ('pickup_committed', 'onboard'):
                    visited[0, p] = 1.0  # pickup already committed; skip
                    visited[0, d] = 0.0  # delivery still needed
                elif status == 'delivery_committed':
                    visited[0, p] = 1.0
                    visited[0, d] = 1.0  # both committed; decoder skips
                # 'delivered': covered by the initial all-1 setting above

            return visited

        def _augment_dynamic_features(self, batch, active_requests, full_instance, t_current: float):
            import torch
            n_total = batch['x'].shape[1]
            n_depots = int(full_instance['n_depots'])
            n_req = int(full_instance['n_req'])
            _TIME_SCALE = 180.0
            device = batch['x'].device

            order_age_col = torch.zeros(1, n_total, device=device)
            del_slack_col = torch.zeros(1, n_total, device=device)

            tw = batch['time_window']  # [1, n_total]
            for req in active_requests.values():
                p_idx = int(req.pickup_node)
                d_idx = int(req.delivery_node)
                t_arr = float(req.t_arrival) if hasattr(req, 't_arrival') else 0.0
                age = ((t_current - t_arr) / _TIME_SCALE)
                age = max(0.0, min(1.0, age))
                order_age_col[0, p_idx] = age

                t_del = float(tw[0, d_idx].item())
                slack = ((t_del - t_current) / _TIME_SCALE)
                slack = max(-1.0, min(1.0, slack))
                del_slack_col[0, d_idx] = slack

            extra = torch.stack([order_age_col, del_slack_col], dim=-1)  # [1, n_total, 2]
            batch = dict(batch)
            batch['x'] = torch.cat([batch['x'], extra], dim=-1)
            return batch

        def solve(self, residual: dict) -> Dict[int, List['Leg']]:
            import torch
            full_instance = residual['full_instance']
            active_requests = residual['active_requests']
            vehicles = residual['vehicles']
            t_current = residual.get('current_time', 0.0)
            n_uav, n_adr = self._num_agents(vehicles)

            batch = self._batchify_instance(full_instance)
            batch = self._mask_unrevealed_nodes(batch, active_requests, full_instance)
            initial_visited = self._build_initial_visited(batch, active_requests, full_instance)

            # If every customer node is pre-visited there is nothing to route.
            # Calling the actor would cause the decoder while-loop to skip entirely,
            # leaving time_log=[] and crashing torch.cat.
            n_depots_fi = int(full_instance['n_depots'])
            if bool(initial_visited[0, n_depots_fi:].all()):
                return {vid: [] for vid in vehicles.keys()}

            if self.dynamic_features:
                batch = self._augment_dynamic_features(
                    batch, active_requests, full_instance, float(t_current)
                )
            with torch.no_grad():
                decoded_tour, _, _ = self.actor(
                    batch,
                    n_uav,
                    n_adr,
                    greedy=True,
                    checkpoint_encoder=False,
                    training=False,
                    initial_visited=initial_visited,
                )

            if decoded_tour.dim() != 3 or decoded_tour.size(0) != 1:
                raise RuntimeError(f'Unexpected actor output shape: {tuple(decoded_tour.shape)}')

            return self._decode_route_plan(
                residual=residual,
                decoded_tour=decoded_tour[0].detach().cpu(),
            )

    return RLSolver(actor, device, dynamic_features=_dynamic_features)

def run_train(rung: str, quick: bool = False, epochs: int = None):
    import VRP_Rollout_train as trainer
    trainer.RUNG = rung
    cfg = trainer.RUNG_CONFIG[rung]
    for k, v in cfg.items():
        setattr(trainer, k, v)
    trainer.n_depots = trainer.n_depots_uav + trainer.n_depots_adr
    trainer.steps = trainer.n_depots + trainer.n_req * 2
    if quick:
        trainer._QUICK_MAX_EPOCHS = epochs if epochs is not None else 5
    else:
        trainer._QUICK_MAX_EPOCHS = None
    trainer.train()

def run_dispatch(model_path: str, delta: float, n_episodes: int):
    import numpy as np
    from dispatch_sim import RollingHorizonDispatcher, sample_arrival_stream
    from creat_vrp import create_instance
    from coalition import make_fleet
    from metrics import compute_runtime_stats, compute_feasibility_rate, compute_violation_magnitude, _extract_request_history

    solver = build_solver(model_path)
    dispatcher = RollingHorizonDispatcher(solver, delta_minutes=delta, shift_minutes=120)

    all_logs = []
    for ep in range(n_episodes):
        inst = create_instance(25, 3, 3, 2, 2, np.random.default_rng(ep))
        fleet = make_fleet(n_uav=3, n_adr=3, n_depots_uav=2, n_depots_adr=2)
        arrivals = sample_arrival_stream(120, 20, seed=ep, max_requests=int(inst['n_req']))
        log = dispatcher.run_shift(arrivals, fleet, inst)
        all_logs.append(log)

    feas = compute_feasibility_rate(all_logs)
    rt = compute_runtime_stats(all_logs)
    viol = compute_violation_magnitude(_extract_request_history(all_logs))
    print(
        f'Δ={delta} min | feasibility={feas:.3f} | '
        f'late_delivery={viol["pct_late_delivery"]:.1%} '
        f'(mean {viol["mean_late_delivery_min"]:.1f} min) | '
        f'late_pickup={viol["pct_late_pickup"]:.1%} '
        f'(mean {viol["mean_late_pickup_min"]:.1f} min) | '
        f'median_solver={rt["per_decision_median_s"] * 1000:.1f}ms | '
        f'shift_time={rt["per_shift_mean_s"]:.2f}s'
    )

def run_coalition(model_path: str, n_episodes: int):
    from coalition import run_full_sweep
    from allocation import allocate
    from run_coalition import analyse, _flatten_result, _save_csv
    import os

    solver = build_solver(model_path)
    results = run_full_sweep(solver, n_episodes=n_episodes, delta_minutes=10)
    allocated = [allocate(r) for r in results]

    analyse(results)

    os.makedirs('eval_results', exist_ok=True)
    rows = [_flatten_result(r) for r in results]
    _save_csv(rows, os.path.join('eval_results', 'coalition_rl_sweep.csv'))

    core_count = sum(1 for r in allocated if r.get('core_shapley', False))
    print(f'\nShapley core exists in {core_count}/{len(results)} configs.')

def run_coalition_greedy(n_episodes: int, delta: float, out_dir: str):
    from run_coalition import main as coalition_main
    import sys
    sys.argv = [
        'run_coalition.py',
        '--episodes', str(n_episodes),
        '--delta', str(delta),
        '--out-dir', out_dir,
    ]
    coalition_main()

def run_evaluate(checkpoint: str, rung: str, n_test: int, generalize: bool,
                 ablate_edges: bool, mode_analysis: bool):
    import subprocess
    cmd = [sys.executable, 'evaluate.py', '--checkpoint', checkpoint, '--rung', rung,
           '--n-test', str(n_test)]
    if generalize:
        cmd.append('--generalize')
    if ablate_edges:
        cmd.append('--ablate-edges')
    if mode_analysis:
        cmd.append('--mode-analysis')
    ret = subprocess.run(cmd, cwd='.')
    return ret.returncode

def run_pre_assess(fast: bool):
    import subprocess
    cmd = [sys.executable, 'pre_assess.py']
    if fast:
        cmd.append('--fast')
    ret = subprocess.run(cmd, cwd='.')
    return ret.returncode

def run_benchmark(extra_args: List[str]):
    import subprocess
    cmd = [sys.executable, 'benchmark_rolling_baselines_with_plots.py'] + extra_args
    ret = subprocess.run(cmd, cwd='.')
    return ret.returncode

def run_tests():
    import subprocess
    ret = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'], cwd='.')
    return ret.returncode

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CE-CPDPTW Phase 1')
    parser.add_argument(
        '--mode',
        choices=['train', 'dispatch', 'coalition', 'coalition-greedy',
                 'evaluate', 'pre-assess', 'benchmark', 'test'],
        default='train',
    )
    parser.add_argument('--rung', choices=['A', 'B', 'C', 'D'], default='A',
                        help='Curriculum rung (train/evaluate)')
    parser.add_argument('--model', default='',
                        help='Path to trained model .pt file (dispatch/coalition)')
    parser.add_argument('--checkpoint', default='',
                        help='Path to training checkpoint .pth (evaluate)')
    parser.add_argument('--delta', type=float, default=10.0,
                        help='Rolling-horizon window in minutes')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Episodes for dispatch/coalition sweep')
    parser.add_argument('--n-test', type=int, default=1024,
                        help='Test instances for evaluate mode')
    parser.add_argument('--out-dir', default='eval_results',
                        help='Output directory for coalition-greedy CSV')
    parser.add_argument('--generalize', action='store_true',
                        help='Evaluate on larger problem sizes (evaluate mode)')
    parser.add_argument('--ablate-edges', action='store_true',
                        help='Run edge-feature ablation (evaluate mode)')
    parser.add_argument('--mode-analysis', action='store_true',
                        help='Run UAV vs ADR mode assignment analysis (evaluate mode)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast pre-assess with 50 instances instead of 200')
    parser.add_argument('--quick', action='store_true',
                        help='CPU run: small data (train=12800, batch=64, val=1280); '
                             'use --epochs to set duration (default 5)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epoch count for quick CPU runs')
    parser.add_argument('--benchmark-args', nargs=argparse.REMAINDER, default=[],
                        help='Extra args forwarded verbatim to benchmark_rolling_baselines_with_plots.py')
    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args.rung, quick=args.quick, epochs=args.epochs)
    elif args.mode == 'dispatch':
        run_dispatch(args.model, args.delta, args.episodes)
    elif args.mode == 'coalition':
        run_coalition(args.model, args.episodes)
    elif args.mode == 'coalition-greedy':
        run_coalition_greedy(args.episodes, args.delta, args.out_dir)
    elif args.mode == 'evaluate':
        sys.exit(run_evaluate(
            args.checkpoint or args.model, args.rung, args.n_test,
            args.generalize, args.ablate_edges, args.mode_analysis,
        ))
    elif args.mode == 'pre-assess':
        sys.exit(run_pre_assess(args.fast))
    elif args.mode == 'benchmark':
        sys.exit(run_benchmark(args.benchmark_args))
    elif args.mode == 'test':
        sys.exit(run_tests())
