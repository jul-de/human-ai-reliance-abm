// Bridge from compare.py to the JS model in docs/model.mjs. Prints JSON.
//   node run_js.mjs pooled '<json config>'   pooled summaries from runExperiment
//   node run_js.mjs agents <cases.json>       per-agent results on fixed outcome streams
import { readFileSync } from 'node:fs';
import { runExperiment, simulateAgent, summarise, reliability, T } from '../docs/model.mjs';

const [mode, arg] = process.argv.slice(2);

if (mode === 'pooled') {
  const res = runExperiment(JSON.parse(arg));
  const out = {};
  for (const k of ['neutral', 'elevated']) {
    const { delays, ...s } = res[k].summary;
    out[k] = s;
  }
  console.log(JSON.stringify(out));
} else if (mode === 'agents') {
  const cases = JSON.parse(readFileSync(arg, 'utf8'));
  const agents = cases.map(c => {
    const draw = (t) => ({ outcome: c.outcomes[t], seen: c.seen[t] === 1 });
    const r = simulateAgent(c.alpha, c.theta, c.env, draw);
    const rel = Array.from({ length: T }, (_, t) => reliability(t, c.env));
    return { ...r, reliability: rel };
  });
  const { delays, ...summary } = summarise(agents.map(a => a.switchTrial), agents.map(a => a.exposure));
  console.log(JSON.stringify({ agents, summary }));
} else {
  throw new Error(`unknown mode ${mode}`);
}
