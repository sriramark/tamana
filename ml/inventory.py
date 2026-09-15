import math
import numpy as np
from scipy import stats as scipy_stats

z_dict = {
    0.80: 0.842,
    0.85: 1.036,
    0.90: 1.282,
    0.92: 1.405,
    0.95: 1.645,
    0.97: 1.881,
    0.98: 2.054,
    0.99: 2.326,
    0.999: 3.090,
}

def get_z_score(service_level: float) -> float:
    if service_level in z_dict:
        return z_dict[service_level]
    return float(scipy_stats.norm.ppf(service_level))

def compute_eoq(annual_demand: float, ordering_cost: float, holding_cost: float) -> dict:
    if holding_cost <= 0 or ordering_cost <= 0 or annual_demand <= 0:
        raise ValueError("Inputs must be greater than zero.")

    num = 2 * annual_demand * ordering_cost
    eoq_val = math.sqrt(num / holding_cost)

    t_ordering = (annual_demand / eoq_val) * ordering_cost
    t_holding = (eoq_val / 2) * holding_cost
    t_cost = t_ordering + t_holding

    return {
        "eoq": round(eoq_val, 2),
        "numerator": round(num, 4),
        "denominator": holding_cost,
        "total_ordering": round(t_ordering, 4),
        "total_holding": round(t_holding, 4),
        "total_cost": round(t_cost, 4),
        "annual_demand": annual_demand,
        "ordering_cost": ordering_cost,
        "holding_cost": holding_cost,
    }

def compute_safety_stock(
    service_level: float,
    demand_std: float,
    lead_time: float
) -> dict:
    z_val = get_z_score(service_level)
    sigma_lt = demand_std * math.sqrt(lead_time)
    ss_val = z_val * sigma_lt

    return {
        "z_score": round(z_val, 4),
        "sigma_lt": round(sigma_lt, 4),
        "safety_stock": round(ss_val, 2),
        "service_level": service_level,
        "demand_std": demand_std,
        "lead_time": lead_time,
        "service_pct": round(service_level * 100, 1),
    }

def compute_reorder_point(
    avg_demand: float,
    lead_time: float,
    safety_stock: float
) -> dict:
    demand_lt = avg_demand * lead_time
    rop_val = demand_lt + safety_stock

    return {
        "rop": round(rop_val, 2),
        "demand_during_lt": round(demand_lt, 4),
        "safety_stock": round(safety_stock, 4),
        "avg_demand": avg_demand,
        "lead_time": lead_time,
    }

def generate_recommendation(
    current_inventory: float,
    rop_val: float,
    eoq_val: float,
    avg_demand: float,
    lead_time: float,
) -> dict:
    deficit = rop_val - current_inventory
    sugg_order = max(eoq_val, eoq_val + deficit)
    periods_left = (current_inventory - rop_val) / avg_demand if avg_demand > 0 else float("inf")

    if current_inventory <= rop_val:
        st = "CRITICAL"
        urg = "danger"
        act = "Place order immediately."
        msg = f"Current stock ({current_inventory:.0f} units) is below Reorder Point ({rop_val:.0f} units)."
    elif current_inventory <= 1.5 * rop_val:
        st = "WARNING"
        urg = "warning"
        act = "Prepare to order soon."
        msg = f"Current stock ({current_inventory:.0f} units) is near Reorder Point ({rop_val:.0f} units)."
    else:
        st = "SAFE"
        urg = "success"
        act = "Stock is sufficient."
        msg = f"Current stock ({current_inventory:.0f} units) is above Reorder Point ({rop_val:.0f} units)."

    return {
        "status": st,
        "urgency": urg,
        "action": act,
        "message": msg,
        "current_inventory": current_inventory,
        "rop": round(rop_val, 2),
        "eoq": round(eoq_val, 2),
        "suggested_order": round(sugg_order, 2),
        "periods_to_rop": round(max(periods_left, 0), 1),
    }
