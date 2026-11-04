"""Check that the browser simulator (docs/model.mjs) matches the Python model.

1. Exact check: the Python reference reproduces the notebook's v4 results (seed 42).
2. Statistical check: for several configurations, the JS port and the Python
   reference are run on many agents each (different RNGs, so the results agree
   in distribution, not digit for digit) and their outcome measures compared.

Run from the repo root:  python validation/compare.py   (needs Node 18+)
"""
import json
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
        js = json.loads(subprocess.check_output(['node', str(HERE / 'run_js.mjs'), json.dumps(cfg)], text=True))
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
    print('\n2. JS simulator vs Python reference (5,000 agents per condition each):')
    b = stat_check()
    print('\nALL CHECKS PASSED' if a and b else '\nSOME CHECKS FAILED')
    sys.exit(0 if a and b else 1)
