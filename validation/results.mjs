// Reproduces the "Results" table on the simulator page (docs/index.html).
// Seed 42, 200 runs x 25 users = 5,000 users per condition.   Usage: node validation/results.mjs
import { runExperiment } from '../docs/model.mjs';

const V3 = { pHigh: 0.9, pLow: 0.55, driftType: 'abrupt', driftWindow: 20, initialTrust: 0.85,
             alphaMean: 0.15, alphaSd: 0.03, thetaNeutral: 0.25, thetaElevated: 0.40, thetaSd: 0.05,
             visibility: 1, nAgents: 25, nRuns: 200, seed: 42 };
const SCENARIOS = [
  ['v2  identical users',              { ...V3, alphaSd: 0, thetaSd: 0 }],
  ['v3  varied users',                 V3],
  ['v4  gradual drift',                { ...V3, driftType: 'gradual' }],
  ['ext half feedback hidden',         { ...V3, visibility: 0.5 }],
  ['ext half hidden + gradual drift',  { ...V3, visibility: 0.5, driftType: 'gradual' }],
];
const f = (x) => x.toFixed(1).padStart(6);
console.log('scenario                          false alarms %   never switch %   mean delay     mean exposure');
console.log('                                    neut   elev     neut   elev     neut   elev    neut   elev');
for (const [name, cfg] of SCENARIOS) {
  const r = runExperiment(cfg);
  const [n, e] = [r.neutral.summary, r.elevated.summary];
  console.log(name.padEnd(32),
    f(100 * n.falseAlarms / n.n), f(100 * e.falseAlarms / e.n), ' ',
    f(100 * (1 - n.switchRate)), f(100 * (1 - e.switchRate)), ' ',
    f(n.meanDelay), f(e.meanDelay), ' ', f(n.meanExposure), f(e.meanExposure));
}
