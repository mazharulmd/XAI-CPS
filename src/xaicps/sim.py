"""PCL-Sim: a closed-loop process-control simulator with known attack targets and
physics-based hazard deadlines.

Process: supply tank T1 -> transfer valve -> heated tank T2 -> centrifugal pump ->
discharge valve -> consumer. Five PI loops:

    LC1  level of T1       <- inlet valve V1
    FC1  transfer flow     <- transfer valve V2
    LC2  level of T2       <- pump speed
    TC1  temperature of T2 <- heater power
    PC1  discharge pressure<- discharge valve V5

Each loop exposes four historian tags (SP, PV, CMD, feedback), plus two measured
disturbances (ENV group). Attacks follow the HAI taxonomy (SP / CV / PV / combined /
short-term) and are applied at the controller level. The *true* physical state is kept
separately from what the controllers and the historian see, so the time until a safety
limit is crossed (the hazard deadline) is known for every attack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

DT = 1.0          # historian / controller sampling period [s]
SUB = 10          # physics sub-steps per sample

LOOPS = {
    "LC1": ["LC1_SP", "LC1_PV", "LC1_CMD", "LC1_POS"],
    "FC1": ["FC1_SP", "FC1_PV", "FC1_CMD", "FC1_POS"],
    "LC2": ["LC2_SP", "LC2_PV", "LC2_CMD", "LC2_SPD"],
    "TC1": ["TC1_SP", "TC1_PV", "TC1_CMD", "TC1_PWR"],
    "PC1": ["PC1_SP", "PC1_PV", "PC1_CMD", "PC1_POS"],
    "ENV": ["ENV_TIN", "ENV_PSUP"],
}
TAGS = [t for v in LOOPS.values() for t in v]
ACTUATOR_TAGS = {t for t in TAGS if t.endswith(("_CMD", "_POS", "_SPD", "_PWR"))}

# Safety limits on the TRUE physical state (hazard if crossed).
LIMITS = {"L1_max": 1.90, "L1_min": 0.10, "L2_max": 1.90, "L2_min": 0.15,
          "T2_max": 85.0, "P_max": 8.5}

# Normal operating ranges for operator setpoint changes.
SP_RANGE = {"LC1": (0.8, 1.3), "FC1": (2.4, 3.6), "LC2": (0.8, 1.3),
            "TC1": (50.0, 62.0), "PC1": (4.0, 5.5)}

PRIMITIVES = ["SP", "CV", "PV_BIAS", "PV_FREEZE", "CV_PV", "SP_PV", "CV_ST"]


@dataclass
class PI:
    kp: float
    ti: float
    lo: float
    hi: float
    direct: float = 1.0          # +1: u increases when PV < SP
    i: float = 0.0

    def step(self, sp, pv, dt=DT):
        e = self.direct * (sp - pv)
        u_un = self.kp * e + self.i
        u = min(max(u_un, self.lo), self.hi)
        if (u == u_un) or (np.sign(e) != np.sign(u_un - u)):     # conditional anti-windup
            self.i += self.kp * dt / self.ti * e
        return u


@dataclass
class Attack:
    loop: str                      # target loop
    primitive: str                 # one of PRIMITIVES
    t_on: int                      # onset (sample index)
    dur: int                       # duration [samples]
    mag: float                     # signed magnitude in engineering units of the manipulated variable
    decoy_loop: str | None = None  # RQ5: small PV bias on an innocent loop
    decoy_mag: float = 0.0

    def active(self, k):
        return self.t_on <= k < self.t_on + self.dur


@dataclass
class Plant:
    rng: np.random.Generator
    A1: float = 1.5
    A2: float = 1.0
    c_in: float = 6.0e-3     # inlet valve coefficient [m3/s per sqrt(bar)] at 100 %
    c12: float = 7.0e-3      # transfer valve coefficient [m3/s per sqrt(m)] at 100 %
    pmax: float = 11.0       # pump shut-off pressure at 100 % speed [bar]
    kq: float = 2.0e5        # pump curve curvature [bar /(m3/s)^2]
    cv5: float = 3.0e-3      # discharge valve coefficient [m3/s per sqrt(bar)]
    tau_v: float = 4.0       # valve time constant [s]
    tau_p: float = 6.0       # pump speed time constant [s]
    tau_h: float = 15.0      # heater time constant [s]
    state: dict = field(default_factory=dict)

    def reset(self):
        self.state = dict(L1=1.0, L2=1.0, T2=55.0, z1=50.0, z2=50.0, n=60.0, q=400.0,
                          z5=50.0, Tin=20.0, Psup=2.0)

    def flows(self, s):
        q_in = self.c_in * s["z1"] / 100 * np.sqrt(max(s["Psup"], 0.0))
        f12 = self.c12 * s["z2"] / 100 * np.sqrt(max(s["L1"], 0.0))
        z = max(s["z5"], 1.0) / 100
        p = self.pmax * (s["n"] / 100) ** 2 / (1 + self.kq * (self.cv5 * z) ** 2)
        p *= min(1.0, max(s["L2"], 0.0) / 0.15)          # suction loss when T2 runs dry
        q_out = self.cv5 * z * np.sqrt(max(p, 0.0))
        return q_in, f12, q_out, p

    def step(self, u, dt=DT):
        """Advance the true physics by one sample with actuator commands u."""
        s = self.state
        h = dt / SUB
        for _ in range(SUB):
            for key, cmd, tau in (("z1", u["LC1"], self.tau_v), ("z2", u["FC1"], self.tau_v),
                                  ("z5", u["PC1"], self.tau_v), ("n", u["LC2"], self.tau_p),
                                  ("q", u["TC1"], self.tau_h)):
                s[key] += h / tau * (cmd - s[key])
            q_in, f12, q_out, p = self.flows(s)
            s["L1"] = max(s["L1"] + h / self.A1 * (q_in - f12), 0.0)
            s["L2"] = max(s["L2"] + h / self.A2 * (f12 - q_out), 0.0)
            vol = max(self.A2 * s["L2"], 0.05)
            dT = (f12 * (s["Tin"] - s["T2"]) / vol
                  + s["q"] * 1e3 / (1000 * 4186 * vol)
                  - 2.0e-4 * (s["T2"] - 20.0))
            s["T2"] += h * dT
        # slow disturbances (random walks with mean reversion)
        s["Tin"] += 0.002 * (20.0 - s["Tin"]) + 0.02 * self.rng.standard_normal()
        s["Psup"] += 0.002 * (2.0 - s["Psup"]) + 0.004 * self.rng.standard_normal()
        q_in, f12, q_out, p = self.flows(s)
        return dict(L1=s["L1"], L2=s["L2"], T2=s["T2"], F12=f12, P=p)


NOISE = {"L1": 0.004, "L2": 0.004, "T2": 0.08, "F12": 0.03, "P": 0.03,
         "z": 0.15, "n": 0.2, "q": 1.5, "Tin": 0.05, "Psup": 0.005}


def simulate(n_steps: int, seed: int, attack: Attack | None = None, sp_change_every=(900, 2700)):
    """Run the closed loop. Returns (X [n,d] historian tags, info dict).

    info["hazard_t"] is the first sample index at which the true state violates LIMITS
    (np.inf if never)."""
    rng = np.random.default_rng(seed)
    plant = Plant(rng)
    plant.reset()
    ctl = {"LC1": PI(60.0, 120.0, 0, 100, +1, 50.0),
           "FC1": PI(8.0, 20.0, 0, 100, +1, 50.0),
           "LC2": PI(80.0, 150.0, 0, 100, -1, 60.0),
           "TC1": PI(25.0, 200.0, 0, 1100, +1, 400.0),
           "PC1": PI(4.0, 25.0, 1, 100, -1, 50.0)}
    sp = {"LC1": 1.0, "FC1": 3.0, "LC2": 1.0, "TC1": 55.0, "PC1": 4.8}
    next_change = {k: int(rng.integers(*sp_change_every)) for k in sp}
    meas = plant.step({k: c.i for k, c in ctl.items()})
    X = np.zeros((n_steps, len(TAGS)), dtype=np.float32)
    true = np.zeros((n_steps, 5), dtype=np.float32)
    hazard_t = np.inf
    frozen = {}
    pv_key = {"LC1": "L1", "FC1": "F12", "LC2": "L2", "TC1": "T2", "PC1": "P"}
    pv_unit = {"LC1": 1e-3 * 1000, "FC1": 1.0, "LC2": 1.0, "TC1": 1.0, "PC1": 1.0}
    st = plant.state
    for k in range(n_steps):
        # operator setpoint schedule (HMI task scheduler, as in HAI)
        for lp in sp:
            if k >= next_change[lp]:
                sp[lp] = float(rng.uniform(*SP_RANGE[lp]))
                next_change[lp] = k + int(rng.integers(*sp_change_every))
        # sensor readings (true + noise); F12 in L/s for readability
        pv_true = {"LC1": meas["L1"], "FC1": meas["F12"] * 1e3, "LC2": meas["L2"],
                   "TC1": meas["T2"], "PC1": meas["P"]}
        pv = {lp: pv_true[lp] + NOISE[pv_key[lp]] * rng.standard_normal() for lp in pv_true}
        sp_used = dict(sp)
        a = attack if (attack is not None and attack.active(k)) else None
        if a is not None:
            lp, prim = a.loop, a.primitive
            if prim in ("SP", "SP_PV"):
                ramp = min(1.0, (k - a.t_on + 1) / 30.0)      # 30 s trapezoid ramp
                sp_used[lp] = sp[lp] + a.mag * ramp
            if prim == "PV_BIAS":
                pv[lp] = pv[lp] + a.mag * min(1.0, (k - a.t_on + 1) / 20.0)
            if prim in ("PV_FREEZE", "CV_PV", "SP_PV"):
                frozen.setdefault(lp, pv[lp])
                pv[lp] = frozen[lp] + 0.5 * NOISE[pv_key[lp]] * rng.standard_normal()
        if attack is not None and attack.decoy_loop and attack.active(k):
            pv[attack.decoy_loop] += attack.decoy_mag
        # controllers
        u = {lp: ctl[lp].step(sp_used[lp], pv[lp]) for lp in ctl}
        if a is not None and a.primitive in ("CV", "CV_PV", "CV_ST"):
            lo, hi = ctl[a.loop].lo, ctl[a.loop].hi
            if a.primitive == "CV_ST":
                on = ((k - a.t_on) % 60) < 15                 # 15 s pulses every 60 s
                if on:
                    u[a.loop] = float(np.clip(u[a.loop] + a.mag, lo, hi))
            else:
                u[a.loop] = float(np.clip(u[a.loop] + a.mag, lo, hi))
        # historian row: what the controllers report
        fb = {"LC1": st["z1"], "FC1": st["z2"], "LC2": st["n"], "TC1": st["q"], "PC1": st["z5"]}
        fbn = {"LC1": NOISE["z"], "FC1": NOISE["z"], "LC2": NOISE["n"], "TC1": NOISE["q"], "PC1": NOISE["z"]}
        row = []
        for lp in ["LC1", "FC1", "LC2", "TC1", "PC1"]:
            row += [sp_used[lp], pv[lp], u[lp], fb[lp] + fbn[lp] * rng.standard_normal()]
        row += [st["Tin"] + NOISE["Tin"] * rng.standard_normal(),
                st["Psup"] + NOISE["Psup"] * rng.standard_normal()]
        X[k] = row
        meas = plant.step(u)
        true[k] = [meas["L1"], meas["L2"], meas["T2"], meas["F12"], meas["P"]]
        if attack is not None and not np.isfinite(hazard_t) and k >= attack.t_on:
            if (meas["L1"] > LIMITS["L1_max"] or meas["L1"] < LIMITS["L1_min"]
                    or meas["L2"] > LIMITS["L2_max"] or meas["L2"] < LIMITS["L2_min"]
                    or meas["T2"] > LIMITS["T2_max"] or meas["P"] > LIMITS["P_max"]):
                hazard_t = k
    return X, {"true": true, "hazard_t": hazard_t}


# Magnitudes per (loop, primitive) in engineering units; 'low' and 'high' levels.
MAGNITUDE = {
    "SP":        {"LC1": 0.85, "FC1": 2.2, "LC2": 0.85, "TC1": 34.0, "PC1": 4.0},
    "CV":        {"LC1": 60.0, "FC1": -45.0, "LC2": -55.0, "TC1": 600.0, "PC1": -50.0},
    "PV_BIAS":   {"LC1": -0.8, "FC1": -2.0, "LC2": 0.8, "TC1": -32.0, "PC1": -3.5},
    "PV_FREEZE": {"LC1": 0.0, "FC1": 0.0, "LC2": 0.0, "TC1": 0.0, "PC1": 0.0},
    "CV_PV":     {"LC1": 60.0, "FC1": -45.0, "LC2": -55.0, "TC1": 600.0, "PC1": -50.0},
    "SP_PV":     {"LC1": 0.85, "FC1": 2.2, "LC2": 0.85, "TC1": 34.0, "PC1": 4.0},
    "CV_ST":     {"LC1": 80.0, "FC1": -60.0, "LC2": -70.0, "TC1": 700.0, "PC1": -70.0},
}


def make_attack_suite(seeds, n_steps=3600, levels=(0.6, 1.0), decoy=False, rng_seed=0):
    """Enumerate test episodes: every loop x primitive x level x seed."""
    rng = np.random.default_rng(rng_seed)
    loops = ["LC1", "FC1", "LC2", "TC1", "PC1"]
    suite = []
    for seed in seeds:
        for lp in loops:
            for prim in PRIMITIVES:
                for lev in levels:
                    t_on = int(rng.integers(1500, 1900))
                    dur = int(rng.integers(300, 900))
                    mag = MAGNITUDE[prim][lp] * lev
                    if prim == "PV_FREEZE":
                        mag = 0.0
                    a = Attack(lp, prim, t_on, dur, mag)
                    if decoy:
                        others = [o for o in loops if o != lp]
                        a.decoy_loop = str(rng.choice(others))
                        a.decoy_mag = {"LC1": 0.03, "FC1": 0.12, "LC2": 0.03, "TC1": 0.6, "PC1": 0.12}[a.decoy_loop]
                    suite.append(dict(seed=int(seed), level=lev, attack=a, n_steps=n_steps))
    return suite
