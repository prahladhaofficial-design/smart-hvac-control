"""Compare thermostat, PPO agent and MPC on summer 2017 (June-September), weather never seen in training.

Run:  python src/evaluate.py        (needs models/ppo_hvac.zip from train_ppo.py)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from building_model import Building, ZoneParams, build_schedule, load_weather
from controllers import MPC, PolicyController, Thermostat, simulate
from hvac_env import HVACEnv

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def main():
    b = Building()
    sched = build_schedule(load_weather(year=2017), b)
    ppo = PPO.load(ROOT / "models" / "ppo_hvac")
    # the realistic MPC plans with a WRONG model of the building (parameters off by 10-25 %)
    wrong = Building(p=ZoneParams(C_i=0.8e7, C_m=1.85e8, R_im=1.1 / 3000, R_io=1.15 / 400, R_mo=0.85 / 250))
    controllers = [
        Thermostat(b),
        PolicyController(ppo, HVACEnv, b, sched),
        MPC(wrong, estimate_state=True, forecast_noise_per_step=0.2, seed=1, margin=1.0,  # margin chosen on 2016 data (tune_mpc_margin.py)
            name="MPC realistic (wrong model + noisy forecast + Kalman filter)"),
        MPC(b),
    ]

    summaries, traces = {}, {}
    for c in controllers:
        print("simulating", c.name, "...")
        summaries[c.name], traces[c.name] = simulate(c, b, sched)

    base = summaries[Thermostat.name]
    for name, s in summaries.items():
        s["cost_saving_vs_thermostat_pct"] = 100 * (base["cost"] - s["cost"]) / base["cost"]
        s["energy_saving_vs_thermostat_pct"] = 100 * (base["energy_kwh"] - s["energy_kwh"]) / base["energy_kwh"]
        s["peak_period_energy_cut_pct"] = 100 * (base["peak_period_kwh"] - s["peak_period_kwh"]) / base["peak_period_kwh"]
    ppo_name, mpc_name = PolicyController.name, MPC.name   # share of the IDEAL MPC saving
    gap = base["cost"] - summaries[mpc_name]["cost"]
    captured = 100 * (base["cost"] - summaries[ppo_name]["cost"]) / gap if gap > 0 else float("nan")
    out = {
        "test_period": "2017-06-01 to 2017-09-30 (15-minute steps, real outdoor temperature, site 'Fox')",
        "controllers": {k: {m: round(v, 2) for m, v in s.items()} for k, s in summaries.items()},
        "ppo_share_of_mpc_cost_saving_pct": round(captured, 1),
    }
    (RES / "metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(out["controllers"]).T.to_csv(RES / "metrics_table.csv")
    print(json.dumps(out, indent=2))
    plot_bars(summaries)
    plot_days(traces, b)


def plot_bars(summaries):
    names = list(summaries)
    short = ["PI thermostat", "PPO (RL)", "MPC realistic", "MPC ideal"]
    metrics = [("cost", "electricity cost (illustrative units)"),
               ("peak_period_kwh", "energy used in peak-price hours (kWh)"),
               ("energy_kwh", "total cooling electricity (kWh)"),
               ("occupied_discomfort_degC_h", "discomfort while occupied (°C·h)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    colors = ["#999999", "#3b6ea5", "#d98c1f", "#2e8b57"]
    for ax, (m, label) in zip(axes, metrics):
        vals = [round(summaries[n][m], 2) for n in names]   # round away floating-point noise (e.g. 1e-13)
        ax.bar(short, vals, color=colors)
        if max(vals) == 0:
            ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Summer 2017 (unseen weather): 4 months of control of a 500 m² office zone")
    fig.tight_layout(); fig.savefig(RES / "comparison.png", dpi=150); plt.close(fig)


def plot_days(traces, b):
    start, end = "2017-07-11", "2017-07-14"   # three hot weekdays
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw=dict(hspace=0.35))
    colors = {"PI thermostat (fixed setpoint)": "#999999",
              "PPO reinforcement-learning agent": "#3b6ea5",
              "MPC realistic (wrong model + noisy forecast + Kalman filter)": "#d98c1f",
              "MPC (perfect model + forecasts, upper bound)": "#2e8b57"}
    any_tr = next(iter(traces.values())).loc[start:end]
    for ax in axes:
        peak = any_tr["price"] > any_tr["price"].min()
        ax.fill_between(any_tr.index, 0, 1, where=peak, transform=ax.get_xaxis_transform(),
                        color="orange", alpha=0.12, label="peak price")
    ax = axes[0]
    ax.plot(any_tr.index, any_tr["t_out"], color="firebrick", lw=1)
    ax.set_ylabel("outdoor °C")
    ax = axes[1]
    ax.fill_between(any_tr.index, any_tr["low"], any_tr["high"], where=any_tr["occupied"] > 0,
                    color="green", alpha=0.1, label="comfort band (occupied)")
    for name, tr in traces.items():
        t = tr.loc[start:end]
        ax.plot(t.index, t["t_in"], color=colors[name], lw=1.3, label=name)
    ax.set_ylabel("indoor °C"); ax.set_ylim(15, 31)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False)
    ax = axes[2]
    for name, tr in traces.items():
        t = tr.loc[start:end]
        ax.plot(t.index, t["q_cool"] / 1000, color=colors[name], lw=1.1)
    ax.set_ylabel("cooling delivered (kW)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Smart controllers pre-cool the building in cheap hours and coast through the price peak")
    fig.savefig(RES / "three_days.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
