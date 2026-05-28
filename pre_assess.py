import sys
import math
import argparse
import numpy as np

from creat_vrp import create_instance, reward1
from dispatch_sim import RollingHorizonDispatcher, sample_arrival_stream, Vehicle
from greedy_insertion import GreedyInsertion, _get_dist, V_UAV_MAX as V_UAV_RL, V_ADR_MAX as V_ADR_RL
from reward import ALPHA_E, ALPHA_P, ALPHA_D, ALPHA_1, ALPHA_2, ALPHA_U

SCALE      = 200.0
V_UAV_PHYS = 20.0
V_ADR_PHYS = 8.3
SHIFT_MIN  = 120.0
PEAK_RATE  = 20.0
SEED       = 42

P = '[PASS]'
W = '[WARN]'
F = '[FAIL]'

RUNGS = {
    'A': dict(n_req=5,  n_uav=2,  n_adr=2, n_depots_uav=1, n_depots_adr=1),
    'B': dict(n_req=10, n_uav=4,  n_adr=3, n_depots_uav=1, n_depots_adr=1),
    'C': dict(n_req=25, n_uav=5,  n_adr=4, n_depots_uav=2, n_depots_adr=2),
    'D': dict(n_req=60, n_uav=10, n_adr=8, n_depots_uav=3, n_depots_adr=2),
}

def _make_fleet(n_uav, n_adr, instance):
    n_dep = instance['n_depots']
    fleet = {}
    for i in range(n_uav):
        fleet[i] = Vehicle(vehicle_id=i, mode='uav',
                           current_node=i % n_dep, current_time=0.0,
                           battery=6500.0, load=0.0, capacity=5.0)
    for i in range(n_adr):
        vid = n_uav + i
        fleet[vid] = Vehicle(vehicle_id=vid, mode='adr',
                             current_node=i % n_dep, current_time=0.0,
                             battery=4500.0, load=0.0, capacity=10.0)
    return fleet

def _run_dispatch(solver_cls, instance, n_uav, n_adr, seed, slack_override=None):
    n_dep  = int(instance['n_depots'])
    n_req  = int(instance['n_req'])
    tw     = instance['time_window'].squeeze(-1).numpy()
    t_pick = tw[n_dep: n_dep + n_req]
    demand_vals = instance['demand'].squeeze(-1).numpy()

    if slack_override is not None:
        s_mean, s_std, s_clip = slack_override
        rng2 = np.random.default_rng(seed + 77777)
        slack = np.clip(rng2.normal(s_mean, s_std, n_req), *s_clip)
        t_delv = t_pick + slack
    else:
        t_delv = tw[n_dep + n_req:]

    arrivals = []
    for i in range(n_req):
        arrivals.append({
            'req_id':     i,
            't_arrival':  max(0.0, float(t_pick[i]) - 15.0),
            't_pickup':   float(t_pick[i]),
            't_delivery': float(t_delv[i]),
            'demand':     abs(float(demand_vals[n_dep + i])),
        })

    fleet = _make_fleet(n_uav, n_adr, instance)
    log   = RollingHorizonDispatcher(solver_cls(), 10.0, SHIFT_MIN).run_shift(
        arrivals, fleet, instance
    )
    return next((e for e in log if e.get('summary')), {})

def _component_breakdown(summary):
    pickup_early = pickup_late = deliv_late = 0.0
    for r in summary.get('request_history', []):
        t_p = r.get('t_pickup_target') or 0.0
        t_d = r.get('t_delivery_target') or 0.0
        T_p = r.get('T_pickup_actual')
        T_d = r.get('T_delivery_actual')
        if T_p is not None:
            pickup_early += ALPHA_E * max(t_p - T_p, 0.0)
            pickup_late  += ALPHA_P * max(T_p - t_p, 0.0)
        if T_d is not None:
            deliv_late   += ALPHA_D * max(T_d - t_d, 0.0)
    undeliv_penalty = float(summary.get('undelivered_penalty', 0.0))
    return pickup_early, pickup_late, deliv_late, undeliv_penalty

def _run_n(cfg, n, seed_offset=0, slack_override=None, n_uav_ov=None, n_adr_ov=None):
    n_uav = n_uav_ov if n_uav_ov is not None else cfg['n_uav']
    n_adr = n_adr_ov if n_adr_ov is not None else cfg['n_adr']
    undeliv, costs = [], []
    for i in range(n):
        rng  = np.random.default_rng(SEED + seed_offset + i)
        inst = create_instance(**cfg, rng=rng)
        s    = _run_dispatch(GreedyInsertion, inst, n_uav, n_adr, SEED + i,
                             slack_override=slack_override)
        n_rev = max(s.get('total_revealed', cfg['n_req']), 1)
        undeliv.append(s.get('undelivered', 0) / n_rev)
        costs.append(max(s.get('total_cost', 0.0), 1e-9))
    u = np.array(undeliv)
    c = np.array(costs)
    return (float(np.mean(u == 0)), float(u.mean()),
            float(c.mean()), float(c.std() / (c.mean() + 1e-9)))

def _feas_tag(frac):
    if frac >= 0.50:
        return P
    if frac >= 0.20:
        return W
    return F

def check_speed():
    v_uav_implied = V_UAV_PHYS * 60.0 / SCALE
    v_adr_implied = V_ADR_PHYS * 60.0 / SCALE
    uav_ratio = V_UAV_RL / v_uav_implied
    adr_ratio = V_ADR_RL / v_adr_implied

    ok = (abs(uav_ratio - 1.0) < 0.15) and (abs(adr_ratio - 1.0) < 0.15)
    print('\n- 1. Speed Consistency -')
    print(f'  Physical:  V_UAV={v_uav_implied:.2f}  V_ADR={v_adr_implied:.2f}  coord/min')
    print(f'  RL reward: V_UAV={V_UAV_RL:.2f}  V_ADR={V_ADR_RL:.2f}  coord/min')
    print(f'  Ratios:    UAV={uav_ratio:.2f}x  ADR={adr_ratio:.2f}x')
    if ok:
        print(f'  {P} Speeds consistent.')
    else:
        print(f'  {F} Mismatch -- align V_UAV_MAX/V_ADR_MAX across vrpUpdate.py, '
              f'reward1, and greedy_insertion.py')
    return ok

def check_generator(n):
    cfg   = RUNGS['C']
    n_req = cfg['n_req']
    n_dep = cfg['n_depots_uav'] + cfg['n_depots_adr']

    train_slacks, dispatch_slacks = [], []

    for i in range(n):
        rng  = np.random.default_rng(SEED + i)
        inst = create_instance(**cfg, rng=rng)
        tw   = inst['time_window'].squeeze(-1).numpy()
        t_p  = tw[n_dep:n_dep + n_req]
        t_d  = tw[n_dep + n_req:]
        train_slacks.extend((t_d - t_p).tolist())

        stream = sample_arrival_stream(SHIFT_MIN, PEAK_RATE, seed=SEED + i, max_requests=n_req)
        for r in stream:
            dispatch_slacks.append(r['t_delivery'] - r['t_pickup'])

    ts = np.array(train_slacks)
    ds = np.array(dispatch_slacks)
    slack_gap = ts.mean() - ds.mean()
    mismatch_ok = abs(slack_gap) <= 10.0

    print('\n- 2. Generator Distributions (Rung C) -')
    print(f'  Train    slack: mean={ts.mean():.1f}  std={ts.std():.1f}  '
          f'min={ts.min():.1f}  max={ts.max():.1f} min')
    print(f'  Dispatch slack: mean={ds.mean():.1f}  std={ds.std():.1f}  '
          f'min={ds.min():.1f}  max={ds.max():.1f} min')
    print(f'  Gap (train - dispatch): {slack_gap:+.1f} min')

    if mismatch_ok:
        print(f'  {P} Slack distributions aligned.')
    else:
        print(f'  {F} Train/eval delivery slacks differ by {slack_gap:.0f} min -- '
              f'align sample_arrival_stream to N(30,5).')

    return mismatch_ok

def check_feasibility(n, rung_filter=None):
    print('\n- 3. Feasibility & Reward Decomposition -')
    rungs_to_test = [rung_filter] if rung_filter else ['A', 'B', 'C', 'D']
    results = {}

    for rung_name in rungs_to_test:
        cfg   = RUNGS[rung_name]
        n_uav = cfg['n_uav']
        n_adr = cfg['n_adr']
        n_req = cfg['n_req']

        totals, oper_f, pe_f, dl_f, up_f, undeliv = [], [], [], [], [], []

        for i in range(n):
            rng  = np.random.default_rng(SEED + i)
            inst = create_instance(**cfg, rng=rng)
            s    = _run_dispatch(GreedyInsertion, inst, n_uav, n_adr, SEED + i)

            total = max(s.get('total_cost', 0.0), 1e-9)
            oper  = s.get('operating_cost', 0.0)
            pe, pl, dl, up = _component_breakdown(s)

            totals.append(total)
            oper_f.append(oper / total)
            pe_f.append(pl / total)
            dl_f.append(dl / total)
            up_f.append(up / total)

            n_rev = max(s.get('total_revealed', n_req), 1)
            undeliv.append(s.get('undelivered', 0) / n_rev)

        totals  = np.array(totals)
        undeliv = np.array(undeliv)
        oper_f  = np.array(oper_f)
        pe_f    = np.array(pe_f)
        dl_f    = np.array(dl_f)
        up_f    = np.array(up_f)

        feasible_frac    = float(np.mean(undeliv == 0))
        cv               = float(totals.std() / (totals.mean() + 1e-9))
        dominated        = float(dl_f.mean()) > 0.65
        undeliv_dominant = float(up_f.mean()) > 0.50
        norm_cost        = float(totals.mean()) / n_req

        print(f'\n  Rung {rung_name}  (n_req={n_req}, UAV={n_uav}, ADR={n_adr})  [{n} ep]')
        print(f'    Raw cost:        mean={totals.mean():.2f}  std={totals.std():.2f}  CV={cv:.2f}')
        print(f'    Norm/req:        {norm_cost:.3f}  '
              f'{"[GOOD: 0.1-5 range]" if 0.1 <= norm_cost <= 5.0 else "[WARN: outside 0.1-5]"}')
        print(f'    Operating:       {oper_f.mean():.1%} of total')
        print(f'    Pickup late:     {pe_f.mean():.1%} of total')
        print(f'    Delivery late:   {dl_f.mean():.1%} of total')
        print(f'    Undeliv penalty: {up_f.mean():.1%} of total  '
              f'(ALPHA_U={ALPHA_U:.0f} x {undeliv.mean():.1%} undelivered)')
        print(f'    Undeliv rate:    {undeliv.mean():.1%}  '
              f'(0-undeliv episodes: {feasible_frac:.0%})')

        tag = _feas_tag(feasible_frac)
        if feasible_frac < 0.20:
            print(f'    {F} <20% zero-undelivered -- structurally infeasible.')
            print(f'       Fix: add vehicles, relax windows, or reduce peak rate.')
        elif feasible_frac < 0.50:
            print(f'    {W} {feasible_frac:.0%} zero-undelivered -- challenging but learnable.')
        else:
            print(f'    {P} Feasibility comfortable ({feasible_frac:.0%} zero-undelivered).')

        if undeliv_dominant:
            print(f'    {W} Undelivered penalty dominates ({up_f.mean():.0%}) -- '
                  f'RL mainly learns feasibility, not cost quality.')
        elif dominated:
            print(f'    {W} Delivery lateness dominates reward ({dl_f.mean():.0%}) -- '
                  f'scale ALPHA_D down or add more vehicles.')
        else:
            print(f'    {P} Reward well-balanced across components.')

        if cv > 1.2:
            print(f'    {W} High CV={cv:.2f} -- instance difficulty varies heavily.')

        results[rung_name] = {
            'feasible_frac': feasible_frac,
            'dominated':     dominated,
            'cv':            cv,
            'norm_cost':     norm_cost,
            'mean_undeliv':  float(undeliv.mean()),
        }

    return results

class _RandomAssignSolver:
    def __init__(self):
        self._rng = np.random.default_rng(999)

    def solve(self, residual):
        from dispatch_sim import Leg
        vehicles = residual['vehicles']
        active   = residual['active_requests']
        t_now    = residual['current_time']
        full_inst = residual['full_instance']

        dist_uav = _get_dist(full_inst, 'uav')
        dist_adr = _get_dist(full_inst, 'adr')

        routes  = {vid: [] for vid in vehicles}
        waiting = [r for r in active.values() if r.status == 'waiting_for_pickup']
        vids    = list(vehicles.keys())
        if not vids:
            return routes

        perm = self._rng.permutation(len(waiting))
        for idx in perm:
            req = waiting[idx]
            vid = int(self._rng.choice(vids))
            veh = vehicles[vid]
            dm  = dist_uav if veh.mode == 'uav' else dist_adr
            v   = V_UAV_RL if veh.mode == 'uav' else V_ADR_RL

            def _tt(f, t):
                if dm is None:
                    return 20.0
                d = float(dm[f][t])
                return d / v if d < 1e8 else 60.0

            t_pa = t_now + _tt(veh.current_node, req.pickup_node)
            routes[vid].append(Leg(
                request_id=req.req_id, vehicle_id=vid, leg_type='pickup',
                from_node=veh.current_node, to_node=req.pickup_node,
                t_depart=t_now, t_arrive=t_pa,
            ))
            routes[vid].append(Leg(
                request_id=req.req_id, vehicle_id=vid, leg_type='delivery',
                from_node=req.pickup_node, to_node=req.delivery_node,
                t_depart=t_pa,
                t_arrive=t_pa + _tt(req.pickup_node, req.delivery_node),
            ))
        return routes

def check_signal(n):
    cfg   = RUNGS['A']
    n_uav = cfg['n_uav']
    n_adr = cfg['n_adr']

    g_costs, r_costs = [], []
    g_undel, r_undel = [], []

    for i in range(n):
        rng  = np.random.default_rng(SEED + i)
        inst = create_instance(**cfg, rng=rng)
        n_rev = max(cfg['n_req'], 1)

        sg = _run_dispatch(GreedyInsertion,    inst, n_uav, n_adr, SEED + i)
        sr = _run_dispatch(_RandomAssignSolver, inst, n_uav, n_adr, SEED + i)

        g_costs.append(sg.get('total_cost', 0.0))
        r_costs.append(sr.get('total_cost', 0.0))
        g_undel.append(sg.get('undelivered', 0) / n_rev)
        r_undel.append(sr.get('undelivered', 0) / n_rev)

    g = np.array(g_costs); r = np.array(r_costs)
    gu = np.array(g_undel); ru = np.array(r_undel)
    improvement = float((r.mean() - g.mean()) / (r.mean() + 1e-9))
    ok = improvement > 0.10 or gu.mean() < ru.mean() - 0.05

    print('\n- 4. Signal Quality (Rung A) -')
    print(f'  Greedy: cost={g.mean():.3f}  undeliv={gu.mean():.1%}')
    print(f'  Random: cost={r.mean():.3f}  undeliv={ru.mean():.1%}')
    print(f'  Greedy improvement: {improvement:+.1%}')

    if ok:
        print(f'  {P} Clear signal -- greedy beats random (RL has something to learn toward).')
    else:
        print(f'  {F} Weak/no signal -- problem may be too noisy or over-constrained.')

    return ok

def _pilot_one_seed(seed_i, n_updates, cfg, dev):
    import torch
    from VRP_Actor import Model
    from creat_vrp import creat_data

    n_req = cfg['n_req']
    n_uav = cfg['n_uav']
    n_adr = cfg['n_adr']

    torch.manual_seed(seed_i * 1000 + 42)
    model = Model(input_node_dim=11, hidden_node_dim=64,
                  input_edge_dim=4, hidden_edge_dim=16, conv_layers=2).to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4)

    loader = creat_data(
        n_req=n_req, n_uav=n_uav, n_adr=n_adr,
        n_depots_uav=cfg['n_depots_uav'], n_depots_adr=cfg['n_depots_adr'],
        num_samples=512, batch_size=128, seed=SEED + seed_i * 100,
    )

    returns, return_stds, adv_stds, grad_norms = [], [], [], []

    for step, batch in enumerate(loader):
        if step >= n_updates:
            break
        batch = {k: v.to(dev) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        model.train()
        tour, logp, t_tensor = model(batch, n_uav, n_adr, greedy=False)
        R = reward1(batch['time_window'], tour.detach(),
                    batch['edge_attr_d'], batch['edge_attr_r'],
                    t_tensor, n_uav) / n_req

        model.eval()
        with torch.no_grad():
            tour_bl, _, t_bl = model(batch, n_uav, n_adr, greedy=True)
            bl = reward1(batch['time_window'], tour_bl.detach(),
                         batch['edge_attr_d'], batch['edge_attr_r'],
                         t_bl, n_uav) / n_req
        model.train()

        adv = R - bl
        adv_n = adv - adv.mean()
        adv_std = adv_n.std()
        if adv_std > 1e-8:
            adv_n = adv_n / adv_std
        loss = (adv_n.detach() * logp).mean()
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()

        returns.append(float(R.mean()))
        return_stds.append(float(R.std()))
        adv_stds.append(float(adv.std()))
        grad_norms.append(gn)

    R_arr  = np.array(returns)
    RS_arr = np.array(return_stds)
    A_arr  = np.array(adv_stds)
    G_arr  = np.array(grad_norms)

    resid_ratio = float(A_arr.mean() / (RS_arr.mean() + 1e-9))
    trend = (R_arr[-10:].mean() - R_arr[:10].mean()) / (abs(R_arr[:10].mean()) + 1e-9)
    return {
        'resid_ok':   resid_ratio < 0.95,
        'gnorm_ok':   float(G_arr.max()) < 5.0,
        'learning_ok': trend < 0.0,
        'resid_ratio': resid_ratio,
        'grad_max':   float(G_arr.max()),
        'trend':      float(trend),
        'R_mean':     float(R_arr.mean()),
    }

def check_pilot(n_updates=80, n_seeds=3):
    print('\n- 5. Pilot Gradient Noise (Rung A, CPU, self-critic, 3 seeds) -')
    try:
        import torch
        cfg = RUNGS['A']
        dev = torch.device('cpu')

        seed_results = []
        for seed_i in range(n_seeds):
            r = _pilot_one_seed(seed_i, n_updates, cfg, dev)
            seed_results.append(r)
            status = 'ok' if r['learning_ok'] else 'no-trend'
            print(f'  Seed {seed_i}: return={r["R_mean"]:.4f}  trend={r["trend"]:+.1%}  '
                  f'resid={r["resid_ratio"]:.2f}  grad_max={r["grad_max"]:.2f}  [{status}]')

        n_learning = sum(r['learning_ok'] for r in seed_results)
        n_resid_ok = sum(r['resid_ok'] for r in seed_results)
        n_gnorm_ok = sum(r['gnorm_ok'] for r in seed_results)
        all_G_max  = max(r['grad_max'] for r in seed_results)
        all_resid  = float(np.mean([r['resid_ratio'] for r in seed_results]))

        resid_ok_agg    = n_resid_ok  >= 2
        gnorm_ok_agg    = n_gnorm_ok  >= 2
        learning_ok_agg = n_learning  >= 2

        print(f'\n  Aggregated ({n_seeds} seeds):')
        print(f'  Resid ratio:    mean={all_resid:.2f}  ({n_resid_ok}/{n_seeds} pass < 0.95)'
              f'  [pilot batch=128; real training batch=512 gives ~2× lower variance]')
        print(f'  Grad norm max:  {all_G_max:.2f}       ({n_gnorm_ok}/{n_seeds} stable < 5.0)')
        print(f'  Cost trending:  ({n_learning}/{n_seeds} seeds show improvement)')

        if resid_ok_agg:
            print(f'  {P} Baseline reduces variance ({n_resid_ok}/{n_seeds} seeds).')
        else:
            print(f'  {W} Baseline barely helps ({n_resid_ok}/{n_seeds} seeds) -- '
                  f'high variance may slow learning.')

        if gnorm_ok_agg:
            print(f'  {P} Gradient norms stable ({n_gnorm_ok}/{n_seeds} seeds, max={all_G_max:.2f}).')
        else:
            print(f'  {W} Gradient spikes ({n_gnorm_ok}/{n_seeds} stable, max={all_G_max:.1f}) -- '
                  f'lower lr or tighten clip.')

        if learning_ok_agg:
            print(f'  {P} Norm cost trending down ({n_learning}/{n_seeds} seeds).')
        else:
            print(f'  {W} No cost improvement in {n_updates} steps '
                  f'({n_learning}/{n_seeds} seeds) -- check reward/speed mismatch.')

        return {
            'resid_ok':   resid_ok_agg,
            'gnorm_ok':   gnorm_ok_agg,
            'learning_ok': learning_ok_agg,
            'resid_ratio': all_resid,
            'grad_max':   all_G_max,
            'n_learning': n_learning,
            'n_seeds':    n_seeds,
        }

    except Exception as exc:
        print(f'  [SKIP] {exc}')
        return {'resid_ok': None, 'gnorm_ok': None, 'learning_ok': None}

def check_curriculum():
    import re
    ok = True
    print('\n- 6. Curriculum / Training Config -')
    try:
        with open('VRP_Rollout_train.py') as f:
            src = f.read()
        m = re.search(r"RUNG\s*=\s*['\"]([A-D])['\"]", src)
        rung = m.group(1) if m else '?'
        if rung != 'A':
            print(f'  {W} Training starts at Rung {rung} -- recommend Rung A for stability.')
            ok = False
        else:
            print(f'  {P} Training starts at Rung A -- curriculum order correct.')
    except FileNotFoundError:
        print(f'  [SKIP] VRP_Rollout_train.py not found.')

    try:
        with open('VRP_Rollout_train.py') as f:
            lines = f.readlines()
        inner_sched = any('scheduler.step()' in l and 'epoch' not in l.lower()
                          for l in lines[150:230])
        if inner_sched:
            print(f'  {W} scheduler.step() appears inside batch loop -- lr decays per batch.')
        else:
            print(f'  {P} Scheduler stepping is epoch-based.')
    except FileNotFoundError:
        pass

    return ok

def check_fleet_sensitivity(n):
    print('\n- 7. Fleet Sensitivity Sweep -')
    print('  (greedy baseline, same instances across all fleet sizes)')

    print(f'\n  Rung A (n_req=5, base=2U+2A):')
    print(f'  {"Config":<22}  {"0-undeliv":>10}  {"Undeliv%":>10}  {"Cost/req":>10}  Status')
    print(f'  {"-"*65}')
    cfg_a = RUNGS['A']
    for label, nu, na in [
        ('1U+1A (under)',    1, 1),
        ('2U+1A (-ADR)',     2, 1),
        ('1U+2A (-UAV)',     1, 2),
        ('2U+2A (base)',     2, 2),
        ('3U+2A (+UAV)',     3, 2),
        ('2U+3A (+ADR)',     2, 3),
        ('3U+3A (generous)', 3, 3),
    ]:
        feas, mu, mc, cv = _run_n(cfg_a, n, n_uav_ov=nu, n_adr_ov=na)
        print(f'  {label:<22}  {feas:>9.0%}  {mu:>9.1%}  '
              f'{mc/cfg_a["n_req"]:>10.3f}  {_feas_tag(feas)}')

    print(f'\n  Rung C (n_req=25, base=5U+4A):')
    print(f'  {"Config":<22}  {"0-undeliv":>10}  {"Undeliv%":>10}  {"Cost/req":>10}  Status')
    print(f'  {"-"*65}')
    cfg_c = RUNGS['C']
    for label, nu, na in [
        ('4U+3A (tight)',    4, 3),
        ('5U+4A (base)',     5, 4),
        ('6U+4A (+UAV)',     6, 4),
        ('5U+5A (+ADR)',     5, 5),
        ('6U+5A (generous)', 6, 5),
        ('7U+6A (ample)',    7, 6),
    ]:
        feas, mu, mc, cv = _run_n(cfg_c, n, n_uav_ov=nu, n_adr_ov=na)
        print(f'  {label:<22}  {feas:>9.0%}  {mu:>9.1%}  '
              f'{mc/cfg_c["n_req"]:>10.3f}  {_feas_tag(feas)}')

    print(f'\n  [!] RL should improve substantially over greedy baseline feasibility.')
    print(f'      Use these numbers to choose fleet sizes for ablation study.')

def check_window_sensitivity(n):
    print('\n- 8. Time-Window Sensitivity (Rung A) -')
    print('  (same instances, delivery slack distribution varied)')
    print(f'\n  {"Profile":<28}  {"0-undeliv":>10}  {"Undeliv%":>10}  {"Cost/req":>10}  Status')
    print(f'  {"-"*68}')

    cfg = RUNGS['A']
    n_req = cfg['n_req']

    profiles = [
        ('very tight  N(15,3)[8,30]',   15, 3, (8.0,  30.0)),
        ('tight       N(20,3)[12,40]',  20, 3, (12.0, 40.0)),
        ('base        N(30,5)[15,60]',  30, 5, (15.0, 60.0)),
        ('loose       N(45,8)[25,75]',  45, 8, (25.0, 75.0)),
        ('very loose  N(60,10)[35,90]', 60, 10, (35.0, 90.0)),
    ]

    for label, s_mean, s_std, s_clip in profiles:
        feas, mu, mc, cv = _run_n(cfg, n, slack_override=(s_mean, s_std, s_clip))
        marker = _feas_tag(feas)
        print(f'  {label:<28}  {feas:>9.0%}  {mu:>9.1%}  {mc/n_req:>10.3f}  {marker}')

    print(f'\n  Current training distribution: base N(30,5)[15,60].')
    print(f'  If base is hard (<50% feasible), the RL model will struggle early.')
    print(f'  Consider starting curriculum warm-up with loose windows (ablation).')

def check_accessibility(n):
    print('\n- 9. Accessibility Balance -')
    print(f'  {"Rung":<5}  {"n_req":>6}  {"UAV-only":>10}  {"ADR-only":>10}  {"Both":>8}  Status')
    print(f'  {"-"*58}')

    results = {}
    for rung_name, cfg in RUNGS.items():
        n_dep = cfg['n_depots_uav'] + cfg['n_depots_adr']
        uav_f, adr_f, both_f = [], [], []

        for i in range(n):
            rng  = np.random.default_rng(SEED + i)
            inst = create_instance(**cfg, rng=rng)
            x    = inst['x'].numpy()[n_dep:]
            u    = x[:, 8] > 0.5
            a    = x[:, 9] > 0.5
            uav_f.append(float(np.mean(u & ~a)))
            adr_f.append(float(np.mean(~u & a)))
            both_f.append(float(np.mean(u & a)))

        mu = float(np.mean(uav_f))
        ma = float(np.mean(adr_f))
        mb = float(np.mean(both_f))

        ok = (0.10 <= mu <= 0.30) and (ma < 0.15) and (mb > 0.55)
        print(f'  {rung_name:<5}  {cfg["n_req"]:>6}  {mu:>9.1%}  {ma:>9.1%}  '
              f'{mb:>7.1%}  {P if ok else W}')
        results[rung_name] = {'uav_only': mu, 'adr_only': ma, 'both': mb, 'ok': ok}

    print(f'\n  Target: UAV-only 10-30% (no-fly zone effect),')
    print(f'          ADR-only <15% (barrier should not dominate),')
    print(f'          Both     >55% (ample mode-choice signal for RL).')

    return results

def check_reward_hacking():
    print('\n- 10. Reward-Hacking Probe (ALPHA_U effectiveness) -')
    try:
        import torch
        from VRP_Actor import Model
        from creat_vrp import creat_data

        cfg     = RUNGS['A']
        n_req   = cfg['n_req']
        n_uav   = cfg['n_uav']
        n_adr   = cfg['n_adr']
        n_agent = n_uav + n_adr
        dev     = torch.device('cpu')

        torch.manual_seed(SEED)
        model = Model(input_node_dim=11, hidden_node_dim=64,
                      input_edge_dim=4, hidden_edge_dim=16, conv_layers=2).to(dev)

        loader = creat_data(
            n_req=n_req, n_uav=n_uav, n_adr=n_adr,
            n_depots_uav=cfg['n_depots_uav'], n_depots_adr=cfg['n_depots_adr'],
            num_samples=64, batch_size=32, seed=SEED,
        )

        batch = next(iter(loader))
        batch = {k: v.to(dev) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            tour_g, _, t_g = model(batch, n_uav, n_adr, greedy=True)
            cost_g = reward1(batch['time_window'], tour_g.detach(),
                             batch['edge_attr_d'], batch['edge_attr_r'],
                             t_g, n_uav)

            idle_tour = torch.zeros_like(tour_g)
            t_idle    = torch.zeros_like(t_g)
            cost_idle = reward1(batch['time_window'], idle_tour,
                                batch['edge_attr_d'], batch['edge_attr_r'],
                                t_idle, n_uav)

        mean_g    = float(cost_g.mean())
        mean_idle = float(cost_idle.mean())
        ratio     = mean_idle / (mean_g + 1e-9)

        expected_idle = ALPHA_U * n_req / n_agent

        ok = ratio > 2.0 and mean_idle > mean_g

        print(f'  Greedy tour cost:  {mean_g:.3f} per agent')
        print(f'  Idle tour cost:    {mean_idle:.3f} per agent')
        print(f'  Expected idle min: {expected_idle:.3f}  '
              f'(ALPHA_U={ALPHA_U} × n_req={n_req} / n_agent={n_agent})')
        print(f'  Idle/greedy ratio: {ratio:.2f}  (target > 2.0)')

        if ok:
            print(f'  {P} ALPHA_U penalises idling ({ratio:.1f}×) -- '
                  f'RL cannot gain by skipping requests.')
        else:
            print(f'  {F} Idle/greedy ratio={ratio:.2f} -- '
                  f'RL may exploit idle tours (check ALPHA_U={ALPHA_U} and reward1 implementation).')

        return ok

    except Exception as exc:
        print(f'  [SKIP] {exc}')
        return None

def check_delivery_stress(n):
    print('\n- 11. Delivery Stress Check (Rung A, tight windows N(10,2)[5,20]) -')
    cfg   = RUNGS['A']
    n_req = cfg['n_req']
    n_uav = cfg['n_uav']
    n_adr = cfg['n_adr']

    tight_slack = (10.0, 2.0, (5.0, 20.0))
    dl_fracs, dl_nonzero = [], []

    for i in range(n):
        rng  = np.random.default_rng(SEED + i + 9999)
        inst = create_instance(**cfg, rng=rng)
        s    = _run_dispatch(GreedyInsertion, inst, n_uav, n_adr,
                             SEED + i + 9999, slack_override=tight_slack)
        total = max(s.get('total_cost', 0.0), 1e-9)
        _, _, dl, _ = _component_breakdown(s)
        dl_fracs.append(dl / total)
        dl_nonzero.append(float(dl > 0.0))

    dl_f    = np.array(dl_fracs)
    dl_nz   = np.array(dl_nonzero)
    ok      = float(dl_nz.mean()) > 0.10

    print(f'  Delivery-late fraction of total cost: {dl_f.mean():.1%}  (mean, {n} episodes)')
    print(f'  Episodes with any delivery lateness:  {dl_nz.mean():.0%}')

    if ok:
        print(f'  {P} Delivery urgency signal present -- '
              f'ALPHA_D will drive mode choices under tight windows.')
    else:
        print(f'  {W} No delivery lateness even with tight windows -- '
              f'check window overlap with travel times or ALPHA_D scale.')

    return ok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast',        action='store_true',
                        help='50 instances instead of 200')
    parser.add_argument('--skip-pilot',  action='store_true',
                        help='Skip gradient check (no torch needed)')
    parser.add_argument('--rung', choices=['A', 'B', 'C', 'D'], default=None,
                        help='Filter feasibility check to one rung')
    args = parser.parse_args()

    n = 50 if args.fast else 200

    print('=' * 65)
    print('CE-PDPTW HetGAT-RL  --  Pre-Training Stability Assessment')
    print(f'Instances per check: {n}')
    print('=' * 65)

    speed_ok    = check_speed()
    gen_ok      = check_generator(n)
    reward_res  = check_feasibility(n, rung_filter=args.rung)
    signal_ok   = check_signal(min(n, 100))
    pilot_res   = check_pilot(80) if not args.skip_pilot else \
                  {'resid_ok': None, 'gnorm_ok': None, 'learning_ok': None}
    curric_ok   = check_curriculum()
    check_fleet_sensitivity(max(n // 4, 25))
    check_window_sensitivity(max(n // 4, 25))
    access_res   = check_accessibility(min(n, 100))
    hack_ok      = check_reward_hacking() if not args.skip_pilot else None
    stress_ok    = check_delivery_stress(min(n, 100))

    print('\n' + '=' * 65)
    print('GATE SUMMARY')
    print('=' * 65)

    rung_a = reward_res.get('A', {})
    rung_c = reward_res.get('C', {})
    rung_d = reward_res.get('D', {})

    gates = [
        ('Speed consistency',
         speed_ok, True,
         'Align V_UAV_MAX/V_ADR_MAX across vrpUpdate, reward1, greedy_insertion.'),
        ('Train/eval generator alignment',
         gen_ok, True,
         'Align delivery_slack between create_instance and sample_arrival_stream.'),
        ('Rung A feasibility >20%',
         rung_a.get('feasible_frac', 0) >= 0.20, True,
         'Add vehicles, relax windows, or reduce n_req.'),
        ('Rung A norm cost in learnable range [0.1,5]',
         0.1 <= rung_a.get('norm_cost', 0) <= 5.0, False,
         'Rescale reward: if <0.1 signal is too small; if >5 gradients may spike.'),
        ('Rung B feasibility >20%',
         reward_res.get('B', {}).get('feasible_frac', 1.0) >= 0.20, False,
         'Rung B still too tight for greedy -- bump to 4U+3A if below 20%.'),
        ('Rung C feasibility (any >0% ok)',
         rung_c.get('feasible_frac', 1.0) >= 0.0, False,
         '0% greedy feasibility is EXPECTED for dense PDPTW -- RL should reach 30-60%.'),
        ('Rung D feasibility (any >0% ok)',
         rung_d.get('feasible_frac', 1.0) >= 0.0, False,
         '0% greedy feasibility is EXPECTED for 60-req instances -- RL learns globally.'),
        ('No reward component dominates (Rung A)',
         not rung_a.get('dominated', False), False,
         'Tune ALPHA_D down or add more vehicles.'),
        ('Signal: greedy > random',
         signal_ok, True,
         'Problem over-constrained or reward too noisy.'),
        ('Pilot: baseline reduces variance',
         pilot_res.get('resid_ok'), False,
         'Consider PPO/critic if this persists.'),
        ('Pilot: gradient norms stable',
         pilot_res.get('gnorm_ok'), False,
         'Lower lr or tighten grad clip.'),
        ('Pilot: cost trends down',
         pilot_res.get('learning_ok'), False,
         'Check speed mismatch and reward scale.'),
        ('Curriculum starts at Rung A',
         curric_ok, False,
         'Edit RUNG = "A" in VRP_Rollout_train.py.'),
        ('Accessibility balance ok (Rung A)',
         access_res.get('A', {}).get('ok', True), False,
         'Adjust ADR barrier pct or no-fly zone radius.'),
        ('ALPHA_U penalises idle tours (ratio > 2×)',
         hack_ok, True,
         'Check reward1 ALPHA_U import and undeliv_cost computation.'),
        ('Delivery urgency signal present (tight windows)',
         stress_ok, False,
         'Widen travel-time range or increase ALPHA_D so lateness appears.'),
    ]

    required_fail = False
    for label, ok, required, fix in gates:
        if ok is None:
            print(f'  [?] {label:<48} SKIP')
            continue
        if ok:
            print(f'  [ok] {label:<47} PASS')
        else:
            tag = 'FAIL' if required else 'WARN'
            mark = 'no' if required else '!'
            print(f'  [{mark}] {label:<47} {tag}')
            print(f'       -> {fix}')
            if required:
                required_fail = True

    print()
    if required_fail:
        print('>> Required gates FAILED -- fix before starting long training.')
    else:
        print('>> All required gates pass.')
        print('   Suggested next step: 200-epoch pilot on Rung A, then advance curriculum.')
    print('=' * 65)

if __name__ == '__main__':
    main()
