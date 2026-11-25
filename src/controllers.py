"""Three controllers that decide how much cooling (Q_cool, watts) to deliver every 15 minutes.

1. Thermostat  - what most buildings do today: fixed setpoint, on/off with a dead-band.
2. PPO agent   - reinforcement learning (trained in train_ppo.py, wrapped here).
3. MPC         - model predictive control: at every step solve a linear program that plans the next
                 24 hours of cooling using the building model + weather/price forecasts, apply the first
                 action, repeat. Two versions:
                   * ideal:     exact model, perfect forecasts, sees the hidden mass temperature -> UPPER BOUND
                   * realistic: model parameters off by 10-25 %, noisy weather forecast (about +/-2 C a day
                                ahead), and the hidden mass temperature estimated by a Kalman filter from
                                indoor-temperature measurements only.
"""
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import linprog

from building_model import DT, STEPS_PER_DAY, Building


class Thermostat:
    """Industry-standard baseline: a PI (proportional-integral) controller tracking a fixed setpoint.

    Occupied hours: setpoint 23.5 C (middle of the 22-25 C comfort band).
    Unoccupied: setback to 28 C (saves energy at night/weekends).
    Starts cooling 2 hours before people arrive ("morning pre-conditioning", standard practice).
    It does NOT know electricity prices or weather forecasts - that is exactly what smarter control adds.
    """
    name = "PI thermostat (fixed setpoint)"

    def __init__(self, b: Building, occ_sp=23.5, unocc_sp=28.0, kp=20000.0, ki=6000.0, preconditioning_h=2.0):
        self.b, self.occ_sp, self.unocc_sp, self.kp, self.ki = b, occ_sp, unocc_sp, kp, ki
        self.pre = int(preconditioning_h * 4)
        self.integral = 0.0

    def act(self, k, x, sched):
        occupied_soon = sched["occupied"][k:k + self.pre + 1].max() > 0
        sp = self.occ_sp if occupied_soon else self.unocc_sp
        return self.track(x, sp)

    def track(self, x, sp):
        """PI law: cooling power needed to pull the indoor temperature to setpoint sp."""
        err = x[0] - sp                      # positive = too warm -> cool
        hours = DT / 3600
        u = self.kp * err + self.ki * (self.integral + err * hours)
        qmax = self.b.p.q_cool_max
        if 0 < u < qmax or (u >= qmax and err < 0) or (u <= 0 and err > 0):
            self.integral += err * hours     # anti-windup: stop integrating when saturated
        return float(np.clip(u, 0, qmax))


class MPC:
    name = "MPC (perfect model + forecasts, upper bound)"

    def __init__(self, b: Building, horizon_steps=STEPS_PER_DAY, comfort_weight=20.0,
                 estimate_state=False, forecast_noise_per_step=0.0, seed=0, name=None, margin=0.0):
        self.b, self.N, self.lam = b, horizon_steps, comfort_weight
        self.estimate_state, self.noise = estimate_state, forecast_noise_per_step
        self.margin = margin   # constraint tightening: plan inside a band shrunk by this much (deg C)
        self.rng = np.random.default_rng(seed)
        if name:
            self.name = name
        self.xhat, self.last_u, self.last_k = None, 0.0, None
        if estimate_state:
            # steady-state Kalman filter: we measure only T_i; T_m must be inferred from how T_i behaves
            Cm = np.array([[1.0, 0.0]])
            Q = np.diag([0.05 ** 2, 0.02 ** 2])   # model-error size per step (deg C)
            R = np.array([[0.1 ** 2]])           # thermometer noise (deg C)
            P = solve_discrete_are(b.A.T, Cm.T, Q, R)
            self.K = (P @ Cm.T @ np.linalg.inv(Cm @ P @ Cm.T + R)).ravel()
        A, B = b.A, b.B
        # impulse responses: how u_j changes T_i at step k+1  ->  lower-triangular matrix G
        N = horizon_steps
        powers = [np.eye(2)]
        for _ in range(N):
            powers.append(A @ powers[-1])
        self.powers = powers
        G = np.zeros((N, N))
        for k in range(N):
            for j in range(k + 1):
                G[k, j] = (powers[k - j] @ B)[0, 0]
        self.G = G

    def _state(self, k, x, sched):
        if not self.estimate_state:
            return x                                       # ideal: sees the true [T_i, T_m]
        y = x[0] + self.rng.normal(0, 0.1)                 # noisy indoor thermometer, T_m never measured
        if self.xhat is None:
            self.xhat = np.array([y, y])
        else:                                              # predict with OUR (imperfect) model, then correct
            kp = self.last_k
            self.xhat = self.b.step(self.xhat, self.last_u, sched["t_out"][kp], sched["q_gain"][kp])
        self.xhat = self.xhat + self.K * (y - self.xhat[0])
        return self.xhat

    def act(self, k, x, sched):
        u = self._plan(k, self._state(k, x, sched), sched)
        self.last_u, self.last_k = u, k
        return u

    def _plan(self, k, x, sched):
        b, N = self.b, self.N
        t_out = sched["t_out"][k:k + N].copy()
        if self.noise > 0 and len(t_out) > 1:              # forecast error grows with look-ahead time
            t_out[1:] += np.cumsum(self.rng.normal(0, self.noise, len(t_out) - 1))
        n = len(t_out)
        if n == 0:
            return 0.0
        q_gain = sched["q_gain"][k:k + n]
        price, cop = sched["price"][k:k + n], sched["cop"][k:k + n]
        low = sched["low"][k + 1:k + n + 1] + self.margin
        high = sched["high"][k + 1:k + n + 1] - self.margin
        if len(low) < n:  # end of data: horizon shorter than one step ahead -> shrink horizon
            n = len(low)
            if n == 0:
                return 0.0
            t_out, q_gain, price, cop = t_out[:n], q_gain[:n], price[:n], cop[:n]
        # free response of T_i (no cooling) over the horizon
        free = np.zeros(n)
        xs = x.copy()
        for i in range(n):
            xs = b.A @ xs + b.E @ np.array([t_out[i], q_gain[i]])
            free[i] = xs[0]
        G = self.G[:n, :n]
        hours = DT / 3600
        # variables z = [u_0..u_{n-1}, s_0..s_{n-1}]   (u in W of cooling, s = comfort violation in deg C)
        c = np.concatenate([price / cop * hours / 1000, self.lam * hours * np.ones(n)])
        I = np.eye(n)
        # predicted indoor temperature is T_i = free + G u  (G < 0 because cooling lowers T_i)
        #   T_i <= high + s   ->   G u - s <= high - free
        #   T_i >= low  - s   ->  -G u - s <= free - low
        A_ub = np.block([[G, -I], [-G, -I]])
        b_ub = np.concatenate([high - free, free - low])
        bounds = [(0, b.p.q_cool_max)] * n + [(0, None)] * n
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        return float(res.x[0]) if res.success else b.p.q_cool_max


class PolicyController:
    """Wraps a trained stable-baselines3 policy so it can be run in the same evaluation loop.
    Every 15 min the agent shifts the thermostat's setpoint; the same PI loop then tracks it."""
    name = "PPO reinforcement-learning agent"

    def __init__(self, model, env_cls, b, sched):
        self.model, self.helper = model, env_cls(b, sched)

        self.helper.hist = None

    def act(self, k, x, sched):
        h = self.helper
        if h.hist is None:                     # first step: no history yet
            h.hist, h.last_a = np.full(4, x[0]), 0
            h.pi.integral = 0.0
        obs = h.observe(k, x)
        a, _ = self.model.predict(obs, deterministic=True)
        h.hist = np.roll(h.hist, 1); h.hist[0] = x[0]
        h.last_a = int(a)
        return h.action_to_watts(int(a), x, k)


def simulate(controller, b: Building, sched_df, x0=(27.0, 27.0)):
    """Run a controller over the whole schedule and return per-step results + summary metrics."""
    sched = {c: sched_df[c].to_numpy() for c in sched_df.columns}
    n = len(sched_df)
    x = np.array(x0, dtype=float)
    t_in, q, kwh, cost, viol = (np.zeros(n) for _ in range(5))
    hours = DT / 3600
    for k in range(n):
        u = float(np.clip(controller.act(k, x, sched), 0, b.p.q_cool_max))
        x = b.step(x, u, sched["t_out"][k], sched["q_gain"][k])
        t_in[k], q[k] = x[0], u
        kwh[k] = u / sched["cop"][k] * hours / 1000
        cost[k] = kwh[k] * sched["price"][k]
        # x is now the state at time k+1, so check it against the comfort band of time k+1
        kn = min(k + 1, n - 1)
        viol[k] = max(0.0, x[0] - sched["high"][kn]) + max(0.0, sched["low"][kn] - x[0])
    occ = np.append(sched["occupied"][1:], sched["occupied"][-1]) > 0   # occupancy at time k+1
    elec_kw = kwh / hours
    summary = dict(
        energy_kwh=float(kwh.sum()),
        cost=float(cost.sum()),
        peak_period_kwh=float(kwh[sched["price"] > sched["price"].min()].sum()),
        max_demand_kw=float(elec_kw.max()),
        occupied_discomfort_degC_h=float(viol[occ].sum() * hours),
        pct_occupied_time_outside_band=float(100 * (viol[occ] > 0.05).mean()),
    )
    trace = sched_df.assign(t_in=t_in, q_cool=q, kwh=kwh, cost=cost, viol=viol)
    return summary, trace
