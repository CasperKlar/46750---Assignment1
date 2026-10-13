"""Optional Q3_battery sensitivity analyses.

Call run_sensitivity(data, output_dir) from main.py after the base case.
No changes to the optimization models or JSON files are required.
"""
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model_Q3_bat import FlexibleConsumerModel as BatteryModel


PRICE_FACTORS = (0.0, 0.5, 1.0, 1.5, 2.0)
CAPACITIES_KWH = np.arange(2.0, 11.0, 1.0)


def _solve(data):
    return BatteryModel(data).build().solve()


def _battery_losses(result, data):
    """Total energy lost in battery conversions, in kWh (one-hour intervals)."""
    h = result.hourly
    return float(
        ((1.0 - data.battery_charging_efficiency) * h["P_ch"]
         + (1.0 / data.battery_discharging_efficiency - 1.0) * h["P_dis"]).sum()
    )

def _plot_price(df, path):
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()
    x = df["alpha"]

    ax1.plot(x, -df["net_utility"], "o-", color="C0", label="Total daily cost")
    ax2.plot(x, df["disutility"], "s-",color="C3", label="Consumption disutility")

    ax1.set(xlabel="Electricity-price spread factor α", ylabel="Total daily cost [DKK/day]", title="Sensitivity to electricity-price variation")
    ax2.set_ylabel("Consumption disutility [DKK/day]")
    ax1.set_xticks(x)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_capacity(df, path):
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()
    x = df["capacity_kWh"]
    ax1.plot(x, -df["net_utility"], "o-", color="C0", label="Total daily cost")
    ax2.plot(x, df["battery_losses_kWh"], "s-", color="C3",
             label="Battery energy losses")
    ax1.set(xlabel="Battery capacity [kWh]", ylabel="Total daily cost [DKK/day]",
            title="Sensitivity to battery capacity")
    ax2.set_ylabel("Battery energy losses [kWh/day]")
    ax1.set_xticks(x)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    

def _plot_price_profiles(data, path):
    """Plot hourly electricity prices for each spread factor."""

    base_prices = np.asarray(data.energy_price, dtype=float)
    mean_price = float(base_prices.mean())
    hours = np.asarray(data.hours)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for alpha in PRICE_FACTORS:
        varied_prices = mean_price + alpha * (base_prices - mean_price)

        ax.step(hours, varied_prices, where="mid", linewidth=2.2 if alpha == 1.0 else 1.5, color="C3" if alpha == 1.0 
            else {0.0: "C0",
                  0.5: "C1",
                  1.5: "C2",
                  2.0: "C4"
            }[alpha], label=fr"$\alpha = {alpha:g}$")

    ax.set(xlabel="Hour", ylabel="Electricity price [DKK/kWh]", title="Electricity-price profiles for different spread factors")

    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    

def run_sensitivity(data, out):
    """Run price-spread and capacity studies; save figures and CSV summaries."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if data.battery_capacity_kWh is None:
        raise ValueError("Sensitivity analysis requires a battery case")

    # -------------------------------------------------------
    # 1. Electricity-price sensitivity (battery only)
    # -------------------------------------------------------
    base_prices = np.asarray(data.energy_price, dtype=float)
    mean_price = float(base_prices.mean())
    _plot_price_profiles(data, out / "sensitivity_price_profiles.png")

    price_rows = []

    for alpha in PRICE_FACTORS:
        varied_prices = mean_price + alpha * (base_prices - mean_price)

        if np.any(varied_prices < 0):
            raise ValueError(f"alpha={alpha} produces negative prices")

        scenario = replace(data, energy_price=varied_prices.copy())
        result = _solve(scenario)

        price_rows.append({"alpha": alpha, "net_utility": result.objective, "disutility": result.disutility,})

    price_df = pd.DataFrame(price_rows)
    price_df.to_csv(out / "sensitivity_price.csv", index=False)
    _plot_price(price_df, out / "sensitivity_price.png")

    # -------------------------------------------------------
    # 2. Battery-capacity sensitivity (unchanged)
    # -------------------------------------------------------
    capacity_rows = []

    for capacity in CAPACITIES_KWH:
        scenario = replace(data, battery_capacity_kWh=capacity, battery_initial_soc_kWh=2.0)

        result = _solve(scenario)

        capacity_rows.append({
            "capacity_kWh": capacity,
            "net_utility": result.objective,
            "battery_losses_kWh": _battery_losses(result, scenario),
            "disutility": result.disutility,
            "simultaneous_hours": int((
                (result.hourly["P_ch"] > 1e-6)
                & (result.hourly["P_dis"] > 1e-6)
            ).sum()),
        })

    capacity_df = pd.DataFrame(capacity_rows)
    capacity_df.to_csv(out / "sensitivity_capacity.csv", index=False)
    _plot_capacity(capacity_df, out / "sensitivity_capacity.png")

    print("\nPrice sensitivity:\n", price_df.round(3).to_string(index=False))
    print("\nCapacity sensitivity:\n", capacity_df.round(3).to_string(index=False))

    return price_df, capacity_df
