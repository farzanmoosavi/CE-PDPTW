import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, Set
import time
import numpy as np

from reward import request_penalty, operating_cost_per_minute, battery_penalty, ALPHA_U
from energy import uav_edge_energy, adr_edge_energy

@dataclass
class Request:
    req_id: int
    t_arrival: float
    t_pickup: float
    t_delivery: float
    pickup_node: int
    delivery_node: int
    demand: float
    status: str = 'unrevealed'
    assigned_vehicle: Optional[int] = None
    T_pickup_actual: Optional[float] = None
    T_delivery_actual: Optional[float] = None

    VALID_TRANSITIONS = {
        'unrevealed': {'waiting_for_pickup'},
        'waiting_for_pickup': {'pickup_committed'},
        'pickup_committed': {'onboard'},
        'onboard': {'delivery_committed'},
        'delivery_committed': {'delivered'},
        'delivered': set(),
    }

    def transition(self, new_status: str) -> None:
        assert new_status in self.VALID_TRANSITIONS[self.status], (
            f'Invalid transition {self.status} -> {new_status} for req {self.req_id}'
        )
        self.status = new_status

@dataclass
class Leg:
    request_id: int
    vehicle_id: int
    leg_type: str
    from_node: int
    to_node: int
    t_depart: float
    t_arrive: float
    travel_time: float = 0.0
    operating_cost: float = 0.0
    t_arrive_raw: float = 0.0
    energy_used: float = 0.0
    t_arrive_committed: Optional[float] = None

@dataclass
class Vehicle:
    vehicle_id: int
    mode: str
    current_node: int
    current_time: float
    battery: float
    load: float
    capacity: float
    committed_leg: Optional[Leg] = None
    onboard_requests: Set[int] = field(default_factory=set)
    battery_init: Optional[float] = None

def _get_distance_matrix(full_instance: Dict[str, Any], mode: str) -> np.ndarray:
    if mode == 'uav':
        if 'edge_attr_d' in full_instance:
            n_total = int(round(np.sqrt(full_instance['edge_attr_d'].numel())))
            return full_instance['edge_attr_d'].view(n_total, n_total).cpu().numpy()
        if 'd_uav' in full_instance:
            return full_instance['d_uav']
    else:
        if 'edge_attr_r' in full_instance:
            n_total = int(round(np.sqrt(full_instance['edge_attr_r'].numel())))
            return full_instance['edge_attr_r'].view(n_total, n_total).cpu().numpy()
        if 'd_adr' in full_instance:
            return full_instance['d_adr']
    raise KeyError(f'No distance matrix found for mode={mode}')

def _vehicle_speed(mode: str) -> float:
    from vrpUpdate import V_UAV_MAX, V_ADR_MAX
    return V_UAV_MAX if mode == 'uav' else V_ADR_MAX

from vrpUpdate import V_UAV_DEPOT, V_ADR_DEPOT

def _travel_time(from_node: int, to_node: int, mode: str, full_instance: Dict[str, Any]) -> float:
    dm = _get_distance_matrix(full_instance, mode)
    d = float(dm[from_node, to_node])
    if d >= 1e9:
        return float('inf')
    v = _vehicle_speed(mode)
    return d / max(v, 1e-9)

def _leg_operating_cost(vehicle: Vehicle, travel_time: float) -> float:
    return operating_cost_per_minute(vehicle.mode == 'uav') * travel_time

def _pickup_speed(distance: float, t_now: float, t_target: float, mode: str) -> float:
    from vrpUpdate import V_UAV_MAX, V_ADR_MAX, V_UAV_MIN_PICKUP, V_ADR_MIN_PICKUP
    if mode == 'uav':
        v_min, v_max = V_UAV_MIN_PICKUP, V_UAV_MAX
    else:
        v_min, v_max = V_ADR_MIN_PICKUP, V_ADR_MAX

    slack = max(t_target - t_now, 0.0)
    if slack <= 0.0 or distance <= 1e-9:
        return v_max
    if distance / max(v_max, 1e-9) >= slack:
        return v_max
    return max(v_min, min(v_max, distance / max(slack, 1e-9)))

def _customer_service_time(vehicle: Vehicle) -> float:
    from vrpUpdate import UAV_LAND_TAKEOFF_MIN
    return UAV_LAND_TAKEOFF_MIN if vehicle.mode == 'uav' else 0.0

def _battery_threshold(vehicle: Vehicle) -> float:
    base = vehicle.battery_init if vehicle.battery_init is not None else vehicle.battery
    frac = 0.25 if vehicle.mode == 'uav' else 0.20
    return frac * float(base)

def _recharge_minutes(vehicle: Vehicle) -> float:
    return 10.0 if vehicle.mode == 'uav' else 20.0

def _count_undelivered(requests: Dict[int, Request]) -> int:
    return sum(1 for r in requests.values() if r.status not in ('delivered', 'unrevealed'))

def _count_revealed(requests: Dict[int, Request]) -> int:
    return sum(1 for r in requests.values() if r.status != 'unrevealed')

def _wind_vec(full_instance: Dict[str, Any]) -> np.ndarray:
    wind = full_instance.get('wind')
    if wind is None:
        return np.zeros(2)
    if hasattr(wind, 'cpu'):
        wind = wind.cpu().numpy()
    wind = np.asarray(wind, dtype=np.float64)
    wind_mag, wind_dir = float(wind[0]), float(wind[1])
    return np.array([wind_mag * math.cos(wind_dir), wind_mag * math.sin(wind_dir)])

def advance_time(
    t_start: float,
    t_end: float,
    vehicles: Dict[int, Vehicle],
    full_instance: Dict[str, Any],
    requests: Optional[Dict[int, 'Request']] = None,
) -> Tuple[List[Leg], Dict[int, Vehicle], float, float, float, float]:
    completed_legs: List[Leg] = []
    step_operating_cost = 0.0
    step_battery_penalty = 0.0
    step_operating_cost_uav = 0.0
    step_operating_cost_adr = 0.0
    n_depots = int(full_instance['n_depots'])

    for veh in vehicles.values():
        leg = veh.committed_leg
        if leg is None:
            veh.current_time = max(veh.current_time, t_end)
            continue

        if veh.battery_init is None:
            veh.battery_init = float(veh.battery)

        if leg.t_arrive_committed is not None:
            if leg.t_arrive_committed > t_end:
                veh.current_time = t_end
                continue
            complete_time = leg.t_arrive_committed
        else:
            dm = _get_distance_matrix(full_instance, veh.mode)
            d = float(dm[leg.from_node, leg.to_node])
            if d >= 1e9:
                veh.current_time = max(veh.current_time, t_end)
                continue

            depart_time = max(veh.current_time, leg.t_depart)

            if depart_time >= t_end:
                veh.current_time = t_end
                continue

            travel_speed = _vehicle_speed(veh.mode)
            raw_arrive = depart_time
            complete_time = depart_time

            is_depot_leg = leg.to_node < n_depots
            req_obj = requests.get(leg.request_id) if requests is not None else None

            if is_depot_leg:
                travel_speed = V_UAV_DEPOT if veh.mode == 'uav' else V_ADR_DEPOT
                travel_time = d / max(travel_speed, 1e-9)
                raw_arrive = depart_time + travel_time
                complete_time = raw_arrive
                if leg.from_node >= n_depots:
                    complete_time += _recharge_minutes(veh)
            elif leg.leg_type == 'pickup':
                target = req_obj.t_pickup if req_obj is not None else depart_time
                travel_speed = _pickup_speed(d, depart_time, target, veh.mode)
                travel_time = d / max(travel_speed, 1e-9)
                raw_arrive = depart_time + travel_time
                complete_time = max(raw_arrive, target) + _customer_service_time(veh)
            else:
                travel_time = d / max(travel_speed, 1e-9)
                raw_arrive = depart_time + travel_time
                complete_time = raw_arrive + _customer_service_time(veh)

            payload = 0.0
            if leg.leg_type == 'delivery' and req_obj is not None:
                payload = float(req_obj.demand)

            wind = _wind_vec(full_instance)
            energy_j = (uav_edge_energy(d, payload, wind, travel_speed)
                        if veh.mode == 'uav' else adr_edge_energy(d, payload))

            leg.t_arrive_raw = raw_arrive
            leg.travel_time = travel_time
            leg.operating_cost = _leg_operating_cost(veh, travel_time)
            leg.energy_used = energy_j / 1000.0
            leg.t_arrive_committed = complete_time

            if complete_time > t_end:
                veh.current_time = t_end
                continue

        is_depot_leg = leg.to_node < n_depots
        leg.t_arrive = complete_time
        veh.current_node = leg.to_node
        veh.current_time = complete_time

        if is_depot_leg:
            veh.battery = float(veh.battery_init)
        else:
            veh.battery = max(0.0, veh.battery - leg.energy_used)

        if leg.leg_type == 'pickup':
            veh.onboard_requests.add(leg.request_id)
        elif leg.leg_type == 'delivery':
            veh.onboard_requests.discard(leg.request_id)

        veh.committed_leg = None
        step_operating_cost += leg.operating_cost
        if veh.mode == 'uav':
            step_operating_cost_uav += leg.operating_cost
        else:
            step_operating_cost_adr += leg.operating_cost
        step_battery_penalty += battery_penalty(
            veh.battery,
            min_threshold=_battery_threshold(veh),
        )
        completed_legs.append(leg)

    return completed_legs, vehicles, step_operating_cost, step_battery_penalty, step_operating_cost_uav, step_operating_cost_adr

def sample_arrival_stream(
    shift_minutes: float,
    peak_rate_per_hour: float,
    profile: str = 'uniform',
    seed: Optional[int] = None,
    max_requests: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rate_per_min = peak_rate_per_hour / 60.0
    arrivals: List[Dict[str, Any]] = []
    req_id = 0
    t = 0.0

    while t < shift_minutes:
        inter = rng.exponential(1.0 / max(rate_per_min, 1e-9))
        t += inter
        if t >= shift_minutes:
            break
        if max_requests is not None and req_id >= max_requests:
            break

        t_pickup = t + rng.uniform(10.0, 20.0)
        delivery_slack = float(np.clip(rng.normal(30.0, 5.0), 15.0, 60.0))
        arrivals.append({
            'req_id': req_id,
            't_arrival': float(t),
            't_pickup': float(t_pickup),
            't_delivery': float(t_pickup + delivery_slack),
            'demand': float(rng.uniform(0.5, 3.0)),
        })
        req_id += 1

    return arrivals

def build_residual_instance(
    requests: Dict[int, Request],
    vehicles: Dict[int, Vehicle],
    current_time: float,
    full_instance: Dict[str, Any],
) -> Dict[str, Any]:
    active_requests = {
        rid: req for rid, req in requests.items()
        if req.status not in ('unrevealed', 'delivered')
    }
    return {
        'full_instance': full_instance,
        'active_requests': active_requests,
        'vehicles': vehicles,
        'current_time': current_time,
    }

def pick_first_uncommitted_leg(route: List[Leg], requests: Dict[int, Request]) -> Optional[Leg]:
    min_customer_node = min(
        [min(r.pickup_node, r.delivery_node) for r in requests.values()],
        default=10**9,
    )
    for leg in route:
        if leg.to_node < min_customer_node:
            return leg
        req = requests.get(leg.request_id)
        if req is None:
            continue
        if leg.leg_type == 'pickup' and req.status == 'waiting_for_pickup':
            return leg
        if leg.leg_type == 'delivery' and req.status in ('onboard', 'delivery_committed'):
            return leg
    return None

class RollingHorizonDispatcher:
    def __init__(self, solver, delta_minutes: float, shift_minutes: float = 120,
                 alpha_p_ratio: float = None):
        self.solver = solver
        self.delta = delta_minutes
        self.T = shift_minutes
        self.alpha_p_ratio = alpha_p_ratio

    def run_shift(
        self,
        arrival_stream: List[Dict[str, Any]],
        fleet_init: Dict[int, Vehicle],
        full_instance: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        t = 0.0
        requests: Dict[int, Request] = {}
        vehicles = {k: Vehicle(**dataclasses.asdict(v)) for k, v in fleet_init.items()}
        for veh in vehicles.values():
            if veh.battery_init is None:
                veh.battery_init = float(veh.battery)
        episode_log: List[Dict[str, Any]] = []

        total_operating_cost = 0.0
        total_operating_cost_uav = 0.0
        total_operating_cost_adr = 0.0
        total_penalty_cost = 0.0
        total_battery_penalty = 0.0

        while t < self.T:
            for r in arrival_stream:
                rid = int(r['req_id'])
                if r['t_arrival'] <= t and rid not in requests:
                    requests[rid] = Request(
                        req_id=rid,
                        t_arrival=float(r['t_arrival']),
                        t_pickup=float(r['t_pickup']),
                        t_delivery=float(r['t_delivery']),
                        pickup_node=int(r.get('pickup_node', full_instance['n_depots'] + rid)),
                        delivery_node=int(r.get('delivery_node', full_instance['n_depots'] + full_instance['n_req'] + rid)),
                        demand=float(r['demand']),
                        status='waiting_for_pickup',
                    )

            residual = build_residual_instance(
                requests=requests,
                vehicles=vehicles,
                current_time=t,
                full_instance=full_instance,
            )

            solve_start = time.perf_counter()
            route_plan = self.solver.solve(residual)
            solver_time_s = time.perf_counter() - solve_start

            for vid, route in route_plan.items():
                veh = vehicles[vid]
                if veh.committed_leg is not None:
                    continue

                first_leg = pick_first_uncommitted_leg(route, requests)
                if first_leg is None:
                    continue

                veh.committed_leg = first_leg
                req = requests.get(first_leg.request_id)

                if req is not None:
                    req.assigned_vehicle = vid

                    if first_leg.leg_type == 'pickup' and req.status == 'waiting_for_pickup':
                        req.transition('pickup_committed')
                    elif first_leg.leg_type == 'delivery' and req.status == 'onboard':
                        req.transition('delivery_committed')

            t_next = min(t + self.delta, self.T)
            completed_legs, vehicles, step_operating_cost, step_battery_penalty, step_oc_uav, step_oc_adr = advance_time(
                t, t_next, vehicles, full_instance, requests
            )
            total_operating_cost += step_operating_cost
            total_operating_cost_uav += step_oc_uav
            total_operating_cost_adr += step_oc_adr
            total_battery_penalty += step_battery_penalty

            completed_payload = []
            step_penalty_cost = 0.0

            for leg in completed_legs:
                req = requests.get(leg.request_id)
                veh = vehicles[leg.vehicle_id]

                if req is not None and leg.leg_type == 'pickup':
                    req.T_pickup_actual = leg.t_arrive_raw
                    if req.status == 'pickup_committed':
                        req.transition('onboard')

                elif req is not None and leg.leg_type == 'delivery':
                    req.T_delivery_actual = leg.t_arrive
                    if req.status == 'delivery_committed':
                        req.transition('delivered')

                    if req.T_pickup_actual is None:
                        req.T_pickup_actual = req.t_pickup

                    step_penalty_cost += request_penalty(
                        T_pickup_actual=req.T_pickup_actual,
                        t_p=req.t_pickup,
                        T_delivery_actual=req.T_delivery_actual,
                        t_d=req.t_delivery,
                        alpha_p_ratio=self.alpha_p_ratio,
                    )

                completed_payload.append({
                    'request_id': leg.request_id,
                    'vehicle_id': leg.vehicle_id,
                    'leg_type': leg.leg_type,
                    'from_node': leg.from_node,
                    'to_node': leg.to_node,
                    't_arrive': leg.t_arrive,
                    't_arrive_raw': leg.t_arrive_raw,
                    'travel_time': leg.travel_time,
                    'operating_cost': leg.operating_cost,
                    'energy_used': leg.energy_used,
                })

            total_penalty_cost += step_penalty_cost
            episode_log.append({
                't': t_next,
                'solver_time_s': solver_time_s,
                'completed': completed_payload,
                'waiting': sum(1 for r in requests.values() if r.status == 'waiting_for_pickup'),
                'onboard': sum(1 for r in requests.values() if r.status == 'onboard'),
                'delivered': sum(1 for r in requests.values() if r.status == 'delivered'),
                'operating_cost_step': step_operating_cost,
                'penalty_cost_step': step_penalty_cost,
                'battery_penalty_step': step_battery_penalty,
            })

            t = t_next

        delivered = [r for r in requests.values() if r.status == 'delivered']
        n_revealed = _count_revealed(requests)
        delivery_violations_min = [
            max(r.T_delivery_actual - r.t_delivery, 0.0)
            for r in delivered
            if r.T_delivery_actual is not None
        ]
        pickup_violations_min = [
            max(r.T_pickup_actual - r.t_pickup, 0.0)
            for r in delivered
            if r.T_pickup_actual is not None
        ]
        n_delivery_violated = sum(1 for v in delivery_violations_min if v > 0.0)
        n_pickup_violated = sum(1 for v in pickup_violations_min if v > 0.0)

        episode_log.append({
            'summary': True,
            'total_revealed': n_revealed,
            'total_delivered': len(delivered),
            'undelivered': _count_undelivered(requests),
            'operating_cost': total_operating_cost,
            'operating_cost_uav': total_operating_cost_uav,
            'operating_cost_adr': total_operating_cost_adr,
            'penalty_cost': total_penalty_cost + total_battery_penalty,
            'battery_penalty_cost': total_battery_penalty,
            'undelivered_penalty': ALPHA_U * _count_undelivered(requests),
            'total_cost': (total_operating_cost + total_penalty_cost
                           + total_battery_penalty
                           + ALPHA_U * _count_undelivered(requests)),
            'tw_delivery_violation_count': n_delivery_violated,
            'tw_delivery_violation_rate': n_delivery_violated / max(n_revealed, 1),
            'tw_delivery_violation_mean_min': float(np.mean(delivery_violations_min)) if delivery_violations_min else 0.0,
            'tw_delivery_violation_max_min': float(np.max(delivery_violations_min)) if delivery_violations_min else 0.0,
            'tw_pickup_violation_count': n_pickup_violated,
            'tw_pickup_violation_rate': n_pickup_violated / max(n_revealed, 1),
            'tw_pickup_violation_mean_min': float(np.mean(pickup_violations_min)) if pickup_violations_min else 0.0,
            'request_history': [
                {
                    'req_id': r.req_id,
                    'status': r.status,
                    't_pickup_target': r.t_pickup,
                    't_delivery_target': r.t_delivery,
                    'T_pickup_actual': r.T_pickup_actual,
                    'T_delivery_actual': r.T_delivery_actual,
                    'pickup_violation_min': max(r.T_pickup_actual - r.t_pickup, 0.0) if r.T_pickup_actual is not None else None,
                    'delivery_violation_min': max(r.T_delivery_actual - r.t_delivery, 0.0) if r.T_delivery_actual is not None else None,
                    'assigned_vehicle': r.assigned_vehicle,
                }
                for r in requests.values()
                if r.status != 'unrevealed'
            ],
        })

        return episode_log


















  # For the current project scope, do it in stages:
  #
  # 1. Now: finish Rung A→C training with the offline static data (current approach). This validates the core model.
  # 2. After Rung C converges: add Level 1 (randomised initial state) as a fine-tuning phase — 10–20 extra epochs with
  # T_t ∈ [0, 80], random battery, random agent positions. No new architecture needed, just a new dataset constructor.
  # Training adds ~10–20% time.
  # 3. For deployment: wrap the trained model in a T_window=15 min re-planning loop. The Level 1 fine-tuning ensures
  # the model handles mid-shift starts well.
  #
  # Level 2 (episode simulation) is the right long-term approach but adds significant engineering complexity to the
  # data pipeline. Worth doing once the core model is validated.



