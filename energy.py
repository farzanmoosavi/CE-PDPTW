import numpy as np
import math

RHO = 1.225
M_UAV = 12.0
C_D1 = 1.49
C_D2 = 2.2
A1 = 0.224
A2 = 0.1
ZETA = 1.4
N_ROTORS = 8
NU_UAV = 0.9

C_R_ADR = 0.25
M_ADR = 30.0
G = 9.81
NU_ADR = 0.8

def _uav_power(payload_kg: float, v_ground: float, wind_vec: np.ndarray) -> float:
    psi = 0.0
    v_wx = wind_vec[0]
    v_wy = wind_vec[1]

    vx = v_ground * math.cos(psi) + v_wx
    vz = v_ground * math.sin(psi) + v_wy
    if abs(vx) < 1e-9 and abs(vz) < 1e-9:
        chi = 0.0
    else:
        chi = math.atan2(vz, vx)

    v_a = math.sqrt(max(0.0,
        2 * v_ground ** 2 + 2 * (wind_vec ** 2).sum()
        - 2 * math.sqrt((wind_vec ** 2).sum()) * v_ground
        * math.cos(math.atan2(v_wy, v_wx) - chi)
    ))

    drag = RHO * (C_D1 * A1 + C_D2 * A2) * v_a ** 2 / 2.0
    W = G * (M_UAV + payload_kg)
    T = drag + W
    alpha = math.atan2(drag, W)

    vi_approx = 1.0
    P = T * (v_a * math.sin(alpha) + vi_approx) / NU_UAV
    return max(0.0, P)

def _adr_power(payload_kg: float, v_ground: float) -> float:
    return C_R_ADR * G * (M_ADR + payload_kg) * v_ground / NU_ADR

def _payload_per_node(n: int, n_depots: int, n_req: int,
                      demand_vals: np.ndarray) -> np.ndarray:
    payload = np.zeros(n, dtype=np.float64)
    payload[n_depots:n_depots + n_req] = demand_vals
    payload[n_depots + n_req:n_depots + 2 * n_req] = demand_vals * 0.5
    return payload

UAV_LT_SECS = 120.0

def uav_energy_matrix(d_matrix: np.ndarray,
                       demand_vals: np.ndarray,
                       demand_full: np.ndarray,
                       n_depots: int,
                       n_req: int,
                       wind_vec: np.ndarray,
                       v_uav_max: float) -> np.ndarray:
    n = d_matrix.shape[0]
    payload_at_node = _payload_per_node(n, n_depots, n_req, demand_vals)

    P_per_dest = np.array([_uav_power(float(p), v_uav_max, wind_vec)
                           for p in payload_at_node], dtype=np.float64)

    reachable = (d_matrix < 1e9) & (d_matrix > 0)
    t_travel = np.where(reachable, d_matrix / v_uav_max, 0.0)
    travel_energy = t_travel * P_per_dest[np.newaxis, :]

    P_hover = np.array([_uav_power(float(p), 0.0, wind_vec)
                        for p in payload_at_node], dtype=np.float64)
    E_lt = np.zeros(n, dtype=np.float64)
    E_lt[n_depots:] = P_hover[n_depots:] * UAV_LT_SECS

    return (travel_energy + E_lt[np.newaxis, :]).astype(np.float32)

def adr_energy_matrix(d_matrix: np.ndarray,
                       demand_vals: np.ndarray,
                       demand_full: np.ndarray,
                       n_depots: int,
                       n_req: int,
                       v_adr_max: float) -> np.ndarray:
    n = d_matrix.shape[0]
    payload_at_node = _payload_per_node(n, n_depots, n_req, demand_vals)

    P_per_dest = C_R_ADR * G * (M_ADR + payload_at_node) * v_adr_max / NU_ADR

    reachable = (d_matrix < 1e9) & (d_matrix > 0)
    t_travel = np.where(reachable, d_matrix / v_adr_max, 0.0)
    return (t_travel * P_per_dest[np.newaxis, :]).astype(np.float32)

def uav_edge_energy(d: float, payload: float, wind_vec: np.ndarray, v_uav: float) -> float:
    if d < 1e-9 or v_uav < 1e-9:
        return 0.0
    t_travel = d / v_uav
    P = _uav_power(payload, v_uav, wind_vec)
    return P * t_travel

def adr_edge_energy(d: float, payload: float) -> float:
    if d < 1e-9:
        return 0.0
    return C_R_ADR * G * (M_ADR + payload) * d / NU_ADR