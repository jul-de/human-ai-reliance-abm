"""Check that the browser simulator (docs/model.mjs) matches the Python model.

1. Notebook check: the Python reference reproduces the notebook's v4 results (seed 42).
2. Exact check: on fixed outcome streams (random ones plus hand-built edge cases), the
   JS and Python agents give identical switch trials and exposure, the two summarise()
   functions give identical outcome measures, and reliability() matches on every trial.
3. Statistical check: for several configurations, the JS port and the Python
   reference are run on many agents each (different RNGs, so the results agree
   in distribution, not digit for digit) and their outcome measures compared.

Run from the repo root:  python validation/compare.py   (needs Node 18+)
"""
import json
import math
import tempfile
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import reference as ref  # noqa: E402

HERE = pathlib.Path(__file__).parent

# Notebook v4 table, as printed in PB440_Model_v1.ipynb.
NOTEBOOK_V4 = {
    'abrupt_neutral':   (27, 223, 223, 9.4, 10.4),
    'abrupt_elevated':  (4, 227, 246, 18.3, 21.7),
    'gradual_neutral':  (28, 216, 222, 19.8, 21.6),
    'gradual_elevated': (0, 202, 250, 28.2, 33.2),
}

BASE = dict(pHigh=0.90, pLow=0.55, driftType='abrupt', driftWindow=20, initialTrust=0.85,
            visibility=1.0, alphaMean=0.15, alphaSd=0.03, thetaNeutral=0.25,
            thetaElevated=0.40, thetaSd=0.05)
CONFIGS = {
    'v3 heterogeneous, abrupt': {},
    'v4 gradual drift':         {'driftType': 'gradual'},
    'v2 homogeneous agents':    {'alphaSd': 0.0, 'thetaSd': 0.0},
    'extension: visibility 0.5': {'visibility': 0.5},
}
N_RUNS, N_AGENTS = 200, 25   # 5,000 agents per condition per implementation

# Tolerances: about 4 standard errors at this sample size.
TOL = {'false_alarm_rate': 0.02, 'switch_rate': 0.02, 'mean_delay': 0.6, 'mean_exposure': 0.6}


def exact_check():
    got = ref.reproduce_notebook_v4()
    ok = True
    for k, (fa, sw, ar, d, e) in NOTEBOOK_V4.items():
        s = got[k]
        match = (s['false_alarms'], s['switched'], s['at_risk']) == (fa, sw, ar) \
            and round(s['mean_delay'], 1) == d and round(s['mean_exposure'], 1) == e
        ok &= match
        print(f"  {k:18s} {'OK ' if match else 'FAIL'}  FA {s['false_alarms']}/250  "
              f"switched {s['switched']}/{s['at_risk']}  delay {s['mean_delay']:.1f}  exposure {s['mean_exposure']:.1f}")
    return ok


def edge_cases():
    """Hand-built agents that sit exactly on the outcome-definition boundaries."""
    env = dict(pHigh=0.9, pLow=0.55, driftType='abrupt', driftWindow=20, initialTrust=0.85)
    ones, seen = [1] * ref.T, [1] * ref.T

    def first_error_at(k, **kw):
        # alpha 0.99: one error drops trust to ~0.01, far past theta 0.5, so the agent switches at k
        out = ones.copy()
        out[k] = 0
        return dict(alpha=0.99, theta=0.5, env=env, outcomes=out, seen=seen.copy(), **kw)

    cases = {
        'switch at trial 0 (false alarm)':          (first_error_at(0),  (0, 0)),
        'switch at trial 49 (last pre-drift: FA)':  (first_error_at(49), (49, 0)),
        'switch at trial 50 (delay 0, exposure 1)': (first_error_at(50), (50, 1)),
        'switch at trial 99 (delay 49, exp. 50)':   (first_error_at(99), (99, 50)),
        'never switches (exposure censored at 50)': (dict(alpha=0.15, theta=0.25, env=env, outcomes=ones.copy(), seen=seen.copy()), (None, 50)),
    }
    # Trust loss exactly equal to theta must NOT trigger a switch (strict >): 0.5 -> 0.25 is exact in binary.
    tie_env = {**env, 'initialTrust': 0.5}
    out = ones.copy(); out[60] = 0
    cases['loss exactly = theta does not switch'] = (dict(alpha=0.5, theta=0.25, env=tie_env, outcomes=out, seen=seen.copy()), (None, 50))
    # Unseen errors leave trust unchanged, but those trials still count as exposure.
    out = ones.copy(); out[60] = 0
    hidden = seen.copy(); hidden[60] = 0
    cases['unseen error: no update, still exposed'] = (dict(alpha=0.99, theta=0.5, env=env, outcomes=out, seen=hidden), (None, 50))
    return cases


def random_cases(n, rng):
    cases = []
    for _ in range(n):
        drift = 'gradual' if rng.random() < 0.5 else 'abrupt'
        env = dict(pHigh=float(rng.uniform(0.5, 1)), pLow=float(rng.uniform(0, 1)), driftType=drift,
                   driftWindow=int(rng.integers(5, 51)), initialTrust=float(rng.uniform(0.5, 1)))
        vis = float(rng.choice([1.0, rng.uniform(0.1, 1)]))
        out = [int(rng.random() < ref.reliability(t, env['pHigh'], env['pLow'], drift, env['driftWindow'])) for t in range(ref.T)]
        seen = [1 if vis >= 1 else int(rng.random() < vis) for _ in range(ref.T)]
        cases.append(dict(alpha=float(rng.uniform(0.01, 0.99)), theta=float(rng.uniform(0.05, 0.99)),
                          env=env, outcomes=out, seen=seen))
    return cases


def py_agent(c):
    env = dict(p_high=c['env']['pHigh'], p_low=c['env']['pLow'], drift_type=c['env']['driftType'],
               drift_window=c['env']['driftWindow'], initial_trust=c['env']['initialTrust'])
    draw = lambda t, p: (c['outcomes'][t], c['seen'][t] == 1)
    rel = [ref.reliability(t, env['p_high'], env['p_low'], env['drift_type'], env['drift_window']) for t in range(ref.T)]
    s, e = ref.simulate_agent(c['alpha'], c['theta'], env, draw)
    return s, e, rel


def exact_agent_check():
    edges = edge_cases()
    cases = [c for c, _ in edges.values()] + random_cases(5000, np.random.default_rng(7))
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(cases, f)
    js = json.loads(subprocess.check_output(['node', str(HERE / 'run_js.mjs'), 'agents', f.name], text=True))
    pathlib.Path(f.name).unlink()

    ok = True
    py = [py_agent(c) for c in cases]
    for (name, (_, expected)), (s, e, _), j in zip(edges.items(), py, js['agents']):
        match = (s, e) == expected == (j['switchTrial'], j['exposure'])
        ok &= match
        print(f"  {'OK  ' if match else 'FAIL'} {name:42s} python {(s, e)}  js {(j['switchTrial'], j['exposure'])}")

    agent_mismatch = sum((s, e) != (j['switchTrial'], j['exposure']) for (s, e, _), j in zip(py, js['agents']))
    rel_mismatch = sum(any(a != b for a, b in zip(rel, j['reliability'])) for (_, _, rel), j in zip(py, js['agents']))
    ps = ref.summarise([p[0] for p in py], [p[1] for p in py])
    jsum = js['summary']
    pairs = [('false_alarms', 'falseAlarms'), ('at_risk', 'atRisk'), ('switched', 'switched'),
             ('switch_rate', 'switchRate'), ('mean_delay', 'meanDelay'), ('mean_exposure', 'meanExposure')]
    sum_ok = all(math.isclose(ps[a], jsum[b], rel_tol=1e-12) for a, b in pairs)
    ok &= agent_mismatch == 0 and rel_mismatch == 0 and sum_ok
    print(f"  {'OK  ' if agent_mismatch == 0 else 'FAIL'} {len(cases):,} agents: switch trial and exposure identical ({agent_mismatch} mismatches)")
    print(f"  {'OK  ' if rel_mismatch == 0 else 'FAIL'} reliability() identical on every trial ({rel_mismatch} mismatching agents)")
    print(f"  {'OK  ' if sum_ok else 'FAIL'} summarise() identical: " + ', '.join(f"{a} {ps[a]:.4g}" for a, _ in pairs))
    return ok


def measures(s):
    return {'false_alarm_rate': s['false_alarms'] / s['n'] if 'false_alarms' in s else s['falseAlarms'] / s['n'],
            'switch_rate': s.get('switch_rate', s.get('switchRate')),
            'mean_delay': s.get('mean_delay', s.get('meanDelay')),
            'mean_exposure': s.get('mean_exposure', s.get('meanExposure'))}


def stat_check():
    ok = True
    np.random.seed(2026)
    for name, override in CONFIGS.items():
        cfg = {**BASE, **override, 'nRuns': N_RUNS, 'nAgents': N_AGENTS, 'seed': 12345}
        js = json.loads(subprocess.check_output(['node', str(HERE / 'run_js.mjs'), 'pooled', json.dumps(cfg)], text=True))
        env = dict(p_high=cfg['pHigh'], p_low=cfg['pLow'], drift_type=cfg['driftType'],
                   drift_window=cfg['driftWindow'], initial_trust=cfg['initialTrust'],
                   visibility=cfg['visibility'])
        print(f"\n  {name}")
        for cond, theta in [('neutral', cfg['thetaNeutral']), ('elevated', cfg['thetaElevated'])]:
            py = ref.run_pooled(N_RUNS, N_AGENTS, theta, alpha_mean=cfg['alphaMean'],
                                alpha_sd=cfg['alphaSd'], theta_sd=cfg['thetaSd'], **env)
            mp, mj = measures(py), measures(js[cond])
            for m, tol in TOL.items():
                diff = abs(mp[m] - mj[m])
                passed = diff <= tol
                ok &= passed
                print(f"    {cond:9s} {m:17s} python {mp[m]:7.3f}  js {mj[m]:7.3f}  "
                      f"|diff| {diff:.3f} (tol {tol})  {'OK' if passed else 'FAIL'}")
    return ok


if __name__ == '__main__':
    print('1. Python reference reproduces the notebook v4 table (seed 42):')
    a = exact_check()
    print('\n2. JS vs Python, agent by agent on fixed outcome streams:')
    b = exact_agent_check()
    print('\n3. JS simulator vs Python reference, pooled (5,000 agents per condition each):')
    c = stat_check()
    ok = a and b and c
    print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
    sys.exit(0 if ok else 1)
