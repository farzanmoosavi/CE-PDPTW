ALPHA_1 = 0.60
ALPHA_2 = 0.10
ALPHA_E = 0.02
ALPHA_P = 0.10
ALPHA_D = 0.15
ALPHA_U = 10.0
LAMBDA_BATTERY = 1.0

alpha_asymmetry_ratio = ALPHA_P / ALPHA_D

def request_penalty(T_pickup_actual: float, t_p: float,
                    T_delivery_actual: float, t_d: float,
                    alpha_p_ratio: float = None) -> float:
    alpha_p = (alpha_p_ratio * ALPHA_D) if alpha_p_ratio is not None else ALPHA_P
    return (
        ALPHA_E * max(t_p - T_pickup_actual, 0.0)
        + alpha_p * max(T_pickup_actual - t_p, 0.0)
        + ALPHA_D * max(T_delivery_actual - t_d, 0.0)
    )

def operating_cost_per_minute(is_uav: bool) -> float:
    return ALPHA_1 if is_uav else ALPHA_2

def battery_penalty(remaining: float, min_threshold: float) -> float:
    return LAMBDA_BATTERY * max(min_threshold - remaining, 0.0)