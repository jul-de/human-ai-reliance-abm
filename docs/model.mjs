// Trust recalibration model: JavaScript port of the v4 simulation in PB440_Model_v1.ipynb.
//
// Each agent updates trust with a delta rule, T <- T + alpha * (O - T), and stops relying on
// the AI the first time (initialTrust - T) > theta. alpha and theta are drawn per agent from
// clipped normal distributions (SD 0 gives homogeneous agents, as in v1/v2).
//
// Extension (not in the notebook): feedback visibility. Each outcome is observed with
// probability `visibility`; unobserved outcomes leave trust unchanged, but the agent is still
// relying on the AI, so post-drift trials still count as exposure. visibility = 1 is the
// original model.
//
// Timing within a trial: the agent relies on the AI, sees (or misses) the outcome, updates trust,
// then decides whether to switch. A trial counts as exposure if it is post-drift and the agent
// relied on the AI during it, so the trial on which an agent switches counts. For switchers,
// exposure = switching delay + 1; agents who never switch have exposure 50 (censored).
//
// validation/compare.py checks this file against the Python reference, both exactly (per agent,
// on fixed outcome streams) and statistically (pooled outcome measures).

export const T = 100;        // trials per run
export const T_DRIFT = 50;   // trial at which reliability starts to drop

// Small, fast, seedable PRNG (mulberry32).
export function makeRng(seed) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const normal = (mean, sd) => {
    if (sd === 0) return mean;
    if (spare !== null) { const z = spare; spare = null; return mean + sd * z; }
    let u = 0;
    while (u === 0) u = uniform();
    const v = uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return mean + sd * r * Math.cos(2 * Math.PI * v);
  };
  return { uniform, normal };
}

const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// AI reliability at trial t (matches get_reliability in the notebook).
export function reliability(t, env) {
  if (t < T_DRIFT) return env.pHigh;
  if (env.driftType === 'abrupt' || t >= T_DRIFT + env.driftWindow) return env.pLow;
  const progress = (t - T_DRIFT) / env.driftWindow;
  return env.pHigh - progress * (env.pHigh - env.pLow);
}

// One agent. draw(t, p) returns { outcome: 0|1, seen: boolean } for trial t at reliability p.
export function simulateAgent(alpha, theta, env, draw, hist = null) {
  let tr = env.initialTrust;
  let switchTrial = null;
  let exposure = 0;
  for (let t = 0; t < T; t++) {
    const { outcome, seen } = draw(t, reliability(t, env));
    if (seen) tr = tr + alpha * (outcome - tr);
    if (hist) hist[t] = tr;
    if (t >= T_DRIFT && switchTrial === null) exposure += 1;  // relied on the AI this trial
    if (switchTrial === null && (env.initialTrust - tr) > theta) switchTrial = t;  // stops from t + 1
  }
  return { switchTrial, exposure };
}

// One run of one condition. Returns per-agent results; trajectories only if keepTrust.
export function runCondition(nAgents, env, agent, rng, keepTrust = false) {
  const switchTrial = new Array(nAgents).fill(null);
  const exposure = new Array(nAgents).fill(0);
  const trust = keepTrust ? [] : null;
  const draw = (t, p) => {
    const outcome = rng.uniform() < p ? 1 : 0;
    const seen = env.visibility >= 1 || rng.uniform() < env.visibility;
    return { outcome, seen };
  };
  for (let i = 0; i < nAgents; i++) {
    const alpha = clip(rng.normal(agent.alphaMean, agent.alphaSd), 0.01, 0.99);
    const theta = clip(rng.normal(agent.thetaMean, agent.thetaSd), 0.05, 0.99);
    const hist = keepTrust ? new Float32Array(T) : null;
    const r = simulateAgent(alpha, theta, env, draw, hist);
    switchTrial[i] = r.switchTrial;
    exposure[i] = r.exposure;
    if (keepTrust) trust.push(hist);
  }
  return { switchTrial, exposure, trust };
}

// Pooled outcome measures, as defined in the notebook's summarise().
export function summarise(switchTrial, exposure) {
  let falseAlarms = 0, atRisk = 0, switched = 0, delaySum = 0, expSum = 0;
  const delays = [];
  for (let i = 0; i < switchTrial.length; i++) {
    const s = switchTrial[i];
    if (s !== null && s < T_DRIFT) { falseAlarms++; continue; }
    atRisk++;
    expSum += exposure[i];
    if (s !== null) { switched++; delaySum += s - T_DRIFT; delays.push(s - T_DRIFT); }
  }
  return {
    n: switchTrial.length,
    falseAlarms,
    atRisk,
    switched,
    never: atRisk - switched,
    switchRate: atRisk ? switched / atRisk : NaN,
    meanDelay: switched ? delaySum / switched : NaN,
    meanExposure: atRisk ? expSum / atRisk : NaN,
    delays,
  };
}

// Full experiment: two conditions (neutral vs elevated threshold) sharing everything but theta.
// Runs are interleaved per condition like the notebook; the first run's trajectories are kept.
export function runExperiment(cfg) {
  const rng = makeRng(cfg.seed);
  const env = {
    pHigh: cfg.pHigh, pLow: cfg.pLow, driftType: cfg.driftType, driftWindow: cfg.driftWindow,
    initialTrust: cfg.initialTrust, visibility: cfg.visibility,
  };
  const conditions = [
    { key: 'neutral', thetaMean: cfg.thetaNeutral },
    { key: 'elevated', thetaMean: cfg.thetaElevated },
  ];
  const pooled = {};
  for (const c of conditions) pooled[c.key] = { switchTrial: [], exposure: [], trust: null, firstSwitch: null };

  for (let run = 0; run < cfg.nRuns; run++) {
    for (const c of conditions) {
      const agent = { alphaMean: cfg.alphaMean, alphaSd: cfg.alphaSd, thetaMean: c.thetaMean, thetaSd: cfg.thetaSd };
      const r = runCondition(cfg.nAgents, env, agent, rng, run === 0);
      const p = pooled[c.key];
      p.switchTrial.push(...r.switchTrial);
      p.exposure.push(...r.exposure);
      if (run === 0) { p.trust = r.trust; p.firstSwitch = r.switchTrial; }
    }
  }
  const out = {};
  for (const c of conditions) {
    const p = pooled[c.key];
    out[c.key] = { thetaMean: c.thetaMean, summary: summarise(p.switchTrial, p.exposure), trust: p.trust, firstSwitch: p.firstSwitch };
  }
  return out;
}
