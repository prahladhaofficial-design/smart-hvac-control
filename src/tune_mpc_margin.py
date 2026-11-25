"""Choose the realistic MPC's safety margin on TRAINING-year weather (July-August 2016), never on the test year.

Run:  python src/tune_mpc_margin.py
"""
import json
from pathlib import Path

from building_model import Building, ZoneParams, build_schedule, load_weather
from controllers import MPC, Thermostat, simulate

RES = Path(__file__).resolve().parent.parent / "results"


def main():
    b = Building()
    sched = build_schedule(load_weather(year=2016, months=(7, 8)), b)
    wrong = Building(p=ZoneParams(C_i=0.8e7, C_m=1.85e8, R_im=1.1 / 3000, R_io=1.15 / 400, R_mo=0.85 / 250))
    base, _ = simulate(Thermostat(b), b, sched)
    rows = []
    for margin in (0.0, 0.25, 0.5, 0.75, 1.0):
        s, _ = simulate(MPC(wrong, estimate_state=True, forecast_noise_per_step=0.2, seed=1, margin=margin), b, sched)
        rows.append(dict(margin_c=margin,
                         cost_saving_pct=round(100 * (1 - s["cost"] / base["cost"]), 1),
                         occupied_discomfort_degC_h=round(s["occupied_discomfort_degC_h"], 2),
                         pct_occupied_time_outside_band=round(s["pct_occupied_time_outside_band"], 2)))
        print(rows[-1])
    (RES / "mpc_margin_tuning_2016.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
