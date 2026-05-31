# torch is NOT imported at module level — only the RL training functions need it.
# Lazy imports inside each function prevent loading torch in baseline worker
# subprocesses (ce_cpdptw_alns → vrpUpdate), which would conflict with OR-Tools'
# libprotobuf and cause SIGSEGV in spawn subprocesses on CC clusters.

SCALE_M_PER_COORD = 200.0

UAV_CRUISE_MPS = 20.0
ADR_CRUISE_MPS = 8.3

V_UAV_MAX = UAV_CRUISE_MPS * 60.0 / SCALE_M_PER_COORD
V_ADR_MAX = ADR_CRUISE_MPS * 60.0 / SCALE_M_PER_COORD

V_UAV_DEPOT = V_UAV_MAX / 2.0
V_ADR_DEPOT = V_ADR_MAX / 2.0

V_UAV_MIN_PICKUP = 8.0 * 60.0 / SCALE_M_PER_COORD
V_ADR_MIN_PICKUP = 2.0 * 60.0 / SCALE_M_PER_COORD

UAV_LAND_TAKEOFF_MIN = 2.0

def PC_uav(payload_kg, v_ground):
    import torch
    nu = 0.9
    rho = 1.225
    mass = 12.0
    c_d1, c_d2 = 1.49, 2.2
    a1, a2 = 0.224, 0.1
    v_w = 0.0
    v_a = torch.sqrt(torch.clamp(v_ground ** 2 + v_w ** 2, min=1e-8))
    drag = rho * (c_d1 * a1 + c_d2 * a2) * v_a ** 2 / 2.0
    weight = 9.8 * (mass + payload_kg)
    thrust = drag + weight
    alpha = torch.atan2(drag, weight)
    vi = 1.0
    power = thrust * (v_a * torch.sin(alpha) + vi) / nu / 1000.0
    return torch.clamp(power, min=0.0)

def PC_adr(payload_kg, v_ground):
    import torch
    c_r = 0.25
    nu = 0.8
    mass = 30.0
    power = c_r * (mass + payload_kg) * 9.8 * v_ground / nu / 1000.0
    return torch.clamp(power, min=0.0)

def _pickup_speed(distance, t_now, t_target, v_min, v_max):
    import torch
    slack = torch.clamp(t_target - t_now, min=0.0)
    exact_speed = distance / torch.clamp(slack, min=1e-8)
    speed = torch.clamp(exact_speed, min=v_min, max=v_max)
    must_rush = (slack <= 0.0) | (distance / v_max >= slack)
    speed = torch.where(must_rush, torch.full_like(speed, v_max), speed)
    speed = torch.where(distance <= 1e-8, torch.full_like(speed, v_min), speed)
    return speed

def _apply_pickup_updates(
    time_out,
    battery_out,
    base_time,
    base_battery,
    mask,
    distance,
    payload,
    t_target,
    v_min,
    v_max,
    power_fn,
    extra_service_time=0.0,
):
    import torch
    if not mask.any():
        return

    t_now = base_time[mask]
    dist = distance[mask]
    speed = _pickup_speed(dist, t_now, t_target[mask], v_min=v_min, v_max=v_max)
    travel_minutes = dist / speed
    arrival = t_now + travel_minutes
    service_complete = torch.maximum(arrival, t_target[mask]) + extra_service_time

    time_out[mask] = service_complete
    battery_out[mask] = (
        base_battery[mask]
        - travel_minutes * power_fn(payload[mask], speed) * 60.0
    )

def _apply_delivery_updates(
    time_out,
    battery_out,
    base_time,
    base_battery,
    mask,
    distance,
    payload,
    speed,
    power_fn,
    extra_service_time=0.0,
):
    import torch
    if not mask.any():
        return

    dist = distance[mask]
    travel_minutes = dist / speed
    time_out[mask] = base_time[mask] + travel_minutes + extra_service_time
    battery_out[mask] = (
        base_battery[mask]
        - travel_minutes * power_fn(payload[mask], speed * torch.ones_like(dist)) * 60.0
    )

def update_state(demand, time_window, battery, T_t, capacity, E,
                 num_uav, actions, edge_attr_uav, edge_attr_adr, num_depots=None):
    import torch
    previous_indices = actions[-2].squeeze(2)
    current_indices = actions[-1].squeeze(2)

    batch_size, n_agents = current_indices.size()
    n_total = demand.size(1)
    if num_depots is None:
        num_depots = int((demand == 0).sum().item() // batch_size)
    num_pickup = (n_total - num_depots) // 2

    current_time_window = torch.gather(time_window, 1, current_indices)
    batch_u = torch.arange(batch_size, device=demand.device).unsqueeze(1).expand(-1, num_uav)
    batch_a = torch.arange(batch_size, device=demand.device).unsqueeze(1).expand(-1, n_agents - num_uav)

    go_depot = current_indices.lt(num_depots)
    go_pickup = (current_indices >= num_depots) & (current_indices < num_depots + num_pickup)
    go_delivery = current_indices >= (num_depots + num_pickup)

    ea_u_full = edge_attr_uav.view(batch_size, n_total, n_total)
    ea_a_full = edge_attr_adr.view(batch_size, n_total, n_total)
    dis_u = ea_u_full[batch_u, previous_indices[:, :num_uav], current_indices[:, :num_uav]]
    dis_a = ea_a_full[batch_a, previous_indices[:, num_uav:], current_indices[:, num_uav:]]

    _MAX_DIST = 50.0
    dis_u = dis_u.clamp(max=_MAX_DIST)
    dis_a = dis_a.clamp(max=_MAX_DIST)

    base_time = T_t.squeeze(2)
    base_battery = battery.squeeze(2)

    time1 = base_time.clone()
    battery1 = base_battery.clone()

    current_demand = torch.gather(demand, 1, current_indices)
    dynamic_capacity = capacity.squeeze(2) - current_demand

    if go_pickup.any():
        _apply_pickup_updates(
            time1[:, :num_uav],
            battery1[:, :num_uav],
            base_time[:, :num_uav],
            base_battery[:, :num_uav],
            go_pickup[:, :num_uav],
            dis_u,
            current_demand[:, :num_uav],
            current_time_window[:, :num_uav],
            v_min=V_UAV_MIN_PICKUP,
            v_max=V_UAV_MAX,
            power_fn=PC_uav,
            extra_service_time=UAV_LAND_TAKEOFF_MIN,
        )
        _apply_pickup_updates(
            time1[:, num_uav:],
            battery1[:, num_uav:],
            base_time[:, num_uav:],
            base_battery[:, num_uav:],
            go_pickup[:, num_uav:],
            dis_a,
            current_demand[:, num_uav:],
            current_time_window[:, num_uav:],
            v_min=V_ADR_MIN_PICKUP,
            v_max=V_ADR_MAX,
            power_fn=PC_adr,
            extra_service_time=0.0,
        )

    if go_delivery.any():
        _apply_delivery_updates(
            time1[:, :num_uav],
            battery1[:, :num_uav],
            base_time[:, :num_uav],
            base_battery[:, :num_uav],
            go_delivery[:, :num_uav],
            dis_u,
            current_demand[:, :num_uav],
            V_UAV_MAX,
            PC_uav,
            extra_service_time=UAV_LAND_TAKEOFF_MIN,
        )
        _apply_delivery_updates(
            time1[:, num_uav:],
            battery1[:, num_uav:],
            base_time[:, num_uav:],
            base_battery[:, num_uav:],
            go_delivery[:, num_uav:],
            dis_a,
            current_demand[:, num_uav:],
            V_ADR_MAX,
            PC_adr,
            extra_service_time=0.0,
        )

    recharge = (~previous_indices.lt(num_depots)) & go_depot
    if go_depot.any():
        battery1[:, :num_uav][go_depot[:, :num_uav]] = E[0]
        battery1[:, num_uav:][go_depot[:, num_uav:]] = E[1]

        time1[:, :num_uav][go_depot[:, :num_uav]] = (
            base_time[:, :num_uav][go_depot[:, :num_uav]]
            + dis_u[go_depot[:, :num_uav]] / V_UAV_DEPOT
        )
        time1[:, num_uav:][go_depot[:, num_uav:]] = (
            base_time[:, num_uav:][go_depot[:, num_uav:]]
            + dis_a[go_depot[:, num_uav:]] / V_ADR_DEPOT
        )

        if recharge.any():
            time1[:, :num_uav][recharge[:, :num_uav]] += 10.0
            time1[:, num_uav:][recharge[:, num_uav:]] += 20.0

    battery1 = torch.clamp(battery1, min=0.0)
    return dynamic_capacity.detach(), time1.detach(), battery1.detach()

def update_mask(demand, capacity, selected, mask, battery, num_uav, E, i,
                acc_uav=None, acc_adr=None,
                edge_attr_d=None, edge_attr_r=None, num_depots=None):
    import torch
    batch_size, n_agents = selected.size()
    if num_depots is None:
        num_depots = int((demand == 0).sum().item() // batch_size)
    n_total = demand.size(1)
    num_pickup = (n_total - num_depots) // 2

    depot_idx = torch.arange(num_depots, device=demand.device)
    pickup_idx = torch.arange(num_depots, num_depots + num_pickup, device=demand.device)
    corresp_del = num_depots + num_pickup + (pickup_idx - num_depots)

    go_depot = selected.lt(num_depots)

    mask_ = mask.clone()
    mask = mask_.scatter_(2, selected.unsqueeze(-1), 1)
    maskfil = mask.max(dim=1)[0].unsqueeze(1).expand(batch_size, n_agents, n_total)
    mask2 = mask.clone()

    if (~go_depot).any():
        mask2[:, :, :num_depots] = mask2[:, :, :num_depots].masked_fill(
            (~go_depot).unsqueeze(-1), 1
        )

    mask[:, :, depot_idx] = torch.where(go_depot.unsqueeze(-1), 1, mask[:, :, depot_idx])

    unvisited_pickups = (mask2[:, :, pickup_idx] == 0)
    mask[:, :, pickup_idx] *= ~unvisited_pickups

    mask[:, :, num_depots + num_pickup:n_total] = 1
    mask[:, :, corresp_del] = torch.where(mask2[:, :, pickup_idx] == 1, 0, mask[:, :, corresp_del])

    mask = torch.where(maskfil == 1, maskfil, mask)

    demand_exc = demand.unsqueeze(1) > capacity
    final_mask = demand_exc + mask2 + mask

    thresh = torch.cat([
        torch.full((batch_size, num_uav), E[0] * 0.25, device=battery.device),
        torch.full((batch_size, n_agents - num_uav), E[1] * 0.20, device=battery.device),
    ], dim=1)
    low_battery = (battery.squeeze(2) < thresh).unsqueeze(-1)
    if low_battery.any():
        final_mask[:, :, :num_depots] = final_mask[:, :, :num_depots].masked_fill(low_battery, 0)
        final_mask[:, :, num_depots:] = final_mask[:, :, num_depots:].masked_fill(low_battery, 1)

    if acc_uav is not None:
        uav_blocked = (acc_uav.float() < 0.5).clone()
        uav_blocked[:, :num_depots] = False
        final_mask[:, :num_uav, :] = final_mask[:, :num_uav, :].masked_fill(
            uav_blocked.unsqueeze(1).expand(-1, num_uav, -1), 1
        )

    if acc_adr is not None:
        n_adr = n_agents - num_uav
        adr_blocked = (acc_adr.float() < 0.5).clone()
        adr_blocked[:, :num_depots] = False
        final_mask[:, num_uav:, :] = final_mask[:, num_uav:, :].masked_fill(
            adr_blocked.unsqueeze(1).expand(-1, n_adr, -1), 1
        )

    b2d = torch.arange(batch_size, device=demand.device).unsqueeze(1)
    if edge_attr_d is not None:
        ea_d = edge_attr_d.reshape(batch_size, n_total, n_total)
        pos_uav = selected[:, :num_uav]
        d_rows = ea_d[b2d.expand(-1, num_uav), pos_uav, :]
        inf_arcs = d_rows >= 1e9
        inf_arcs[:, :, :num_depots] = False
        final_mask[:, :num_uav, :] = final_mask[:, :num_uav, :].masked_fill(inf_arcs, 1)

    if edge_attr_r is not None:
        ea_r = edge_attr_r.reshape(batch_size, n_total, n_total)
        n_adr = n_agents - num_uav
        pos_adr = selected[:, num_uav:]
        d_rows = ea_r[b2d.expand(-1, n_adr), pos_adr, :]
        inf_arcs = d_rows >= 1e9
        inf_arcs[:, :, :num_depots] = False
        final_mask[:, num_uav:, :] = final_mask[:, num_uav:, :].masked_fill(inf_arcs, 1)

    return final_mask.detach(), mask2.detach()
