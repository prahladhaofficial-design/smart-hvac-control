"""Physics model of one office zone: a "2R2C" thermal network (two temperatures, resistors + capacitors).

    T_i  = indoor air temperature   (air + furniture, heats/cools fast, ~1 hour)
    T_m  = building mass temperature (concrete, walls, slabs: slow, ~13 hours)

Heat balance (watts):
    C_i dT_i/dt = (T_m - T_i)/R_im + (T_o - T_i)/R_io + Q_internal + Q_solar - Q_cool
    C_m dT_m/dt = (T_i - T_m)/R_im + (T_o - T_m)/R_mo

    T_o      outdoor temperature (real measured data)
    Q_cool   heat removed by the air conditioner (our control input, watts of cooling)
    electric power = Q_cool / COP, where COP (coefficient of performance) gets worse when it is hot outside.

Because the equations are linear we can discretise them EXACTLY for a 15-minute step with the matrix
exponential (zero-order hold): x[k+1] = A x[k] + B u[k] + E d[k].
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

DT = 15 * 60  # control step: 15 minutes, in seconds
STEPS_PER_HOUR = 4
STEPS_PER_DAY = 96


@dataclass
class ZoneParams:
    floor_area_m2: float = 500.0
    C_i: float = 1.0e7      # J/K   air + furniture
    C_m: float = 1.5e8      # J/K   structural mass
    R_im: float = 1 / 3000  # K/W   air <-> mass surfaces
    R_io: float = 1 / 400   # K/W   air <-> outside (windows, ventilation, infiltration)
    R_mo: float = 1 / 250   # K/W   mass <-> outside (walls, roof)
    q_internal_occ: float = 25.0    # W/m2 people + lights + computers when occupied
    q_internal_unocc: float = 6.0   # W/m2 standby equipment at night
    solar_peak_w: float = 15000.0   # W peak solar gain through windows (clear day, noon)
    q_cool_max: float = 60000.0     # W maximum cooling capacity of the AC unit
    cop_nominal: float = 4.5        # COP at 25 C outdoor
    cop_slope: float = 0.08         # COP lost per degree C above 25 C


@dataclass
class Comfort:
    occ_low: float = 22.0     # occupied comfort band (deg C), typical summer office range
    occ_high: float = 25.0
    unocc_low: float = 18.0   # loose band at night / weekends
    unocc_high: float = 30.0


@dataclass
class Tariff:
    """Time-of-use electricity price (illustrative numbers, per kWh)."""
    off_peak: float = 0.08
    peak: float = 0.30
    peak_start_h: int = 13
    peak_end_h: int = 19   # peak = 13:00-19:00 on weekdays


@dataclass
class Building:
    p: ZoneParams = field(default_factory=ZoneParams)
    comfort: Comfort = field(default_factory=Comfort)
    tariff: Tariff = field(default_factory=Tariff)

    def __post_init__(self):
        p = self.p
        # continuous-time state space  dx/dt = Ac x + Bc u + Ec d,  x=[T_i, T_m], u=Q_cool, d=[T_o, Q_gain]
        Ac = np.array([
            [-(1 / p.R_im + 1 / p.R_io) / p.C_i, (1 / p.R_im) / p.C_i],
            [(1 / p.R_im) / p.C_m, -(1 / p.R_im + 1 / p.R_mo) / p.C_m],
        ])
        Bc = np.array([[-1 / p.C_i], [0.0]])
        Ec = np.array([[1 / (p.R_io * p.C_i), 1 / p.C_i],
                       [1 / (p.R_mo * p.C_m), 0.0]])
        # exact zero-order-hold discretisation via one big matrix exponential
        n = 2
        M = np.zeros((n + 3, n + 3))
        M[:n, :n] = Ac
        M[:n, n:n + 1] = Bc
        M[:n, n + 1:] = Ec
        Md = expm(M * DT)
        self.A = Md[:n, :n]
        self.B = Md[:n, n:n + 1]
        self.E = Md[:n, n + 1:]

    # ------------------------------------------------------------------ helpers
    def cop(self, t_out):
        p = self.p
        return np.clip(p.cop_nominal - p.cop_slope * (t_out - 25.0), 2.0, 6.0)

    def step(self, x, q_cool, t_out, q_gain):
        return self.A @ x + self.B[:, 0] * q_cool + self.E @ np.array([t_out, q_gain])


# ---------------------------------------------------------------------- exogenous inputs
def load_weather(site="Fox", year=2016, months=(6, 7, 8, 9)) -> pd.DataFrame:
    """Real hourly outdoor temperature from the BDG2 dataset, resampled to 15 minutes.
    Site 'Fox' is a hot, dry US site (summer highs above 40 C)."""
    path = Path(__file__).resolve().parent.parent / "data" / "weather.csv"
    w = pd.read_csv(path, parse_dates=["timestamp"])
    w = w[w["site_id"] == site].set_index("timestamp")["airTemperature"].sort_index()
    w = w[~w.index.duplicated()].asfreq("h").interpolate(limit=12)
    w = w[(w.index.year == year) & (w.index.month.isin(months))]
    w = w.resample("15min").interpolate()
    return w.to_frame("t_out")


def build_schedule(weather: pd.DataFrame, b: Building) -> pd.DataFrame:
    """Add occupancy, internal + solar gains, price, comfort band and COP for every 15-minute step."""
    df = weather.copy()
    idx = df.index
    hour = idx.hour + idx.minute / 60
    weekday = idx.dayofweek < 5
    df["occupied"] = (weekday & (hour >= 8) & (hour < 18)).astype(float)
    p = b.p
    q_int = np.where(df["occupied"] > 0, p.q_internal_occ, p.q_internal_unocc) * p.floor_area_m2
    # simple clear-sky solar shape: half-sine between 6:00 and 20:00
    solar = np.clip(np.sin(np.pi * (hour - 6) / 14), 0, None) * p.solar_peak_w
    df["q_gain"] = q_int + solar
    tf = b.tariff
    peak = weekday & (hour >= tf.peak_start_h) & (hour < tf.peak_end_h)
    df["price"] = np.where(peak, tf.peak, tf.off_peak)
    c = b.comfort
    df["low"] = np.where(df["occupied"] > 0, c.occ_low, c.unocc_low)
    df["high"] = np.where(df["occupied"] > 0, c.occ_high, c.unocc_high)
    df["cop"] = b.cop(df["t_out"].to_numpy())
    df["hour"] = hour
    df["weekday"] = weekday.astype(float)
    return df
