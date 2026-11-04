"""Python reference for the trust recalibration model.

`run_condition` is the notebook's run_simulation_v4 with one addition: feedback
visibility (each outcome is observed with probability `visibility`; unobserved
outcomes leave trust unchanged). With visibility = 1 it makes exactly the same
random draws as the notebook, so `reproduce_notebook_v4()` recovers the
notebook's v4 table digit for digit.
"""
import numpy as np

T = 100
T_DRIFT = 50


def reliability(t, p_high, p_low, drift_type, drift_window):
    if t < T_DRIFT:
        return p_high
    if drift_type == 'abrupt' or t >= T_DRIFT + drift_window:
        return p_low
    progress = (t - T_DRIFT) / drift_window
    return p_high - progress * (p_high - p_low)


def simulate_agent(alpha, theta, env, draw):
    """One agent. draw(t, p) returns (outcome, seen) for trial t at reliability p.

    Timing within a trial: rely on the AI, see (or miss) the outcome, update
    trust, then decide whether to switch. The switch trial itself counts as
    exposure (the agent relied on the AI during it), so for switchers
    exposure = delay + 1.
    """
    trust = env['initial_trust']
    switch_trial = None
    exposure = 0
    for t in range(T):
        outcome, seen = draw(t, reliability(t, env['p_high'], env['p_low'],
                                            env['drift_type'], env['drift_window']))
        if seen:
            trust = trust + alpha * (outcome - trust)
        if t >= T_DRIFT and switch_trial is None:
            exposure += 1
        if switch_trial is None and (env['initial_trust'] - trust) > theta:
            switch_trial = t
    return switch_trial, exposure


def run_condition(n_agents, alpha_mean, alpha_sd, theta_mean, theta_sd,
                  p_high=0.90, p_low=0.55, drift_type='abrupt', drift_window=20,
                  initial_trust=0.85, visibility=1.0):
    env = dict(p_high=p_high, p_low=p_low, drift_type=drift_type,
               drift_window=drift_window, initial_trust=initial_trust)
    alphas = np.random.normal(alpha_mean, alpha_sd, n_agents).clip(0.01, 0.99)
    thetas = np.random.normal(theta_mean, theta_sd, n_agents).clip(0.05, 0.99)

    def draw(t, p):
        outcome = np.random.binomial(1, p)
        seen = visibility >= 1 or np.random.random() < visibility
        return outcome, seen

    switch_trial = [None] * n_agents
    exposure = np.zeros(n_agents)
    for i in range(n_agents):
        switch_trial[i], exposure[i] = simulate_agent(alphas[i], thetas[i], env, draw)
    return switch_trial, exposure


def summarise(switch_trial, exposure):
    false_alarms = sum(1 for s in switch_trial if s is not None and s < T_DRIFT)
    at_risk = [i for i, s in enumerate(switch_trial) if s is None or s >= T_DRIFT]
    switched = [switch_trial[i] for i in at_risk if switch_trial[i] is not None]
    return {
        'n': len(switch_trial),
        'false_alarms': false_alarms,
        'at_risk': len(at_risk),
        'switched': len(switched),
        'switch_rate': len(switched) / len(at_risk) if at_risk else float('nan'),
        'mean_delay': float(np.mean([s - T_DRIFT for s in switched])) if switched else float('nan'),
        'mean_exposure': float(np.mean([exposure[i] for i in at_risk])) if at_risk else float('nan'),
    }


def run_pooled(n_runs, n_agents, theta_mean, alpha_mean=0.15, alpha_sd=0.03, theta_sd=0.05, **env):
    st, ex = [], []
    for _ in range(n_runs):
        s, e = run_condition(n_agents, alpha_mean, alpha_sd, theta_mean, theta_sd, **env)
        st.extend(s)
        ex.extend(e.tolist())
    return summarise(st, ex)


def reproduce_notebook_v4():
    """Re-run the notebook's v4 cell (seed 42, same draw order) and return its table."""
    np.random.seed(42)
    pooled = {f'{d}_{f}': ([], []) for d in ['abrupt', 'gradual'] for f in ['neutral', 'elevated']}
    for _ in range(10):
        for drift_type in ['abrupt', 'gradual']:
            for framing, theta_mean in [('neutral', 0.25), ('elevated', 0.40)]:
                s, e = run_condition(25, 0.15, 0.03, theta_mean, 0.05, drift_type=drift_type)
                pooled[f'{drift_type}_{framing}'][0].extend(s)
                pooled[f'{drift_type}_{framing}'][1].extend(e.tolist())
    return {k: summarise(*v) for k, v in pooled.items()}
