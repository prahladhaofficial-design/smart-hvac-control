"""Gymnasium environment: the RL agent controls the air conditioner of the 2R2C office zone.

Every 15 minutes the agent sees (observation):
    indoor temperature, outdoor temperature now and forecast (+1 h, +3 h, +6 h),
    time of day, weekday flag, occupancy now and soon, electricity price now and soon.
It chooses an OFFSET to the thermostat's usual setpoint schedule (residual, supervisory control), and the
same PI loop the thermostat uses turns that setpoint into cooling power. This is how RL is deployed on real building management systems:
the agent decides *what temperature to aim for when*, the proven low-level loop does the tracking.
Reward = -( electricity cost  +  COMFORT_WEIGHT * degree-hours outside the comfort band ).
So the agent must learn to keep people comfortable while paying as little as possible -
e.g. "pre-cool the concrete in the cheap morning, coast through the expensive afternoon".
The building mass temperature T_m is NOT observed (a real building has no sensor for it).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from building_model import DT, Building

COMFORT_WEIGHT = 20.0   # same weight the MPC uses -> fair comparison
# RESIDUAL control: the agent shifts the thermostat's normal setpoint schedule (23.5 C occupied, 28 C setback)
# by one of these offsets. Offset 0 = behave exactly like the thermostat, so learning starts from a sensible policy.
OFFSETS = np.array([-6.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0])
N_LEVELS = len(OFFSETS)
EPISODE_STEPS = 7 * 96  # one week per episode during training


class HVACEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, b: Building, sched_df, train=True, seed=None):
        super().__init__()
        self.b, self.train = b, train
        self.s = {c: sched_df[c].to_numpy() for c in sched_df.columns}
        self.n = len(sched_df)
        from controllers import Thermostat  # local import avoids a circular import
        self.pi = Thermostat(b)
        self.action_space = spaces.Discrete(N_LEVELS)
        self.observation_space = spaces.Box(-5, 5, shape=(22,), dtype=np.float32)
        self.hist = np.zeros(4)  # indoor temperature over the last hour -> lets the agent infer the hidden mass state
        self.rng = np.random.default_rng(seed)

    # ---- helpers shared with evaluation
    def default_setpoint(self, k):
        soon = self.s["occupied"][k:k + self.pi.pre + 1].max() > 0
        return self.pi.occ_sp if soon else self.pi.unocc_sp

    def action_to_watts(self, a: int, x, k=None) -> float:
        k = self.k if k is None else k
        sp = float(np.clip(self.default_setpoint(k) + OFFSETS[a], 20.0, 30.0))
        return self.pi.track(x, sp)

    def _at(self, key, k):
        return self.s[key][min(k, self.n - 1)]

    def observe(self, k, x, prev_action=None):
        h = self._at("hour", k)
        if prev_action is None:
            prev_action = getattr(self, "last_a", 0)
        obs = [
            (x[0] - 24.0) / 3.0,
            *((x[0] - self.hist) / 1.0),          # change of indoor temp over the last 15/30/45/60 min
            prev_action / (N_LEVELS - 1),
            *[(self._at("t_out", k + d) - 32.0) / 8.0 for d in (0, 4, 12, 24)],
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            self._at("weekday", k),
            *[self._at("occupied", k + d) for d in (0, 4, 8)],
            *[self._at("price", k + d) / 0.30 for d in (0, 8, 16, 24)],
            (self._at("high", k) - x[0]) / 3.0,   # margin to the upper comfort limit
            self.default_setpoint(min(k, self.n - 1)) / 30.0 - 0.85,
        ]
        return np.asarray(obs, dtype=np.float32)

    # ---- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.k = int(self.rng.integers(0, self.n - EPISODE_STEPS - 1))
        self.k0 = self.k
        t0 = self.rng.uniform(23.0, 30.0)
        self.x = np.array([t0, t0 + self.rng.uniform(-1.5, 1.5)])
        self.hist = np.full(4, self.x[0])
        self.last_a = 0
        self.pi.integral = 0.0
        return self.observe(self.k, self.x), {}

    def step(self, action):
        k, s = self.k, self.s
        u = self.action_to_watts(int(action), self.x)
        self.hist = np.roll(self.hist, 1); self.hist[0] = self.x[0]
        self.last_a = int(action)
        self.x = self.b.step(self.x, u, s["t_out"][k], s["q_gain"][k])
        hours = DT / 3600
        kwh = u / s["cop"][k] * hours / 1000
        cost = kwh * s["price"][k]
        kn = min(k + 1, self.n - 1)   # new state belongs to time k+1
        viol = max(0.0, self.x[0] - s["high"][kn]) + max(0.0, s["low"][kn] - self.x[0])
        reward = -(cost + COMFORT_WEIGHT * viol * hours)
        self.k += 1
        truncated = (self.k - self.k0) >= EPISODE_STEPS or self.k >= self.n - 1
        return self.observe(self.k, self.x), float(reward), False, truncated, {"cost": cost, "viol": viol}
