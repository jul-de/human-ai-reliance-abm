// Prints pooled summaries from the JS model (docs/model.mjs) as JSON, for compare.py.
// Usage: node run_js.mjs '<json config>'
import { runExperiment } from '../docs/model.mjs';

const cfg = JSON.parse(process.argv[2]);
const res = runExperiment(cfg);
const out = {};
for (const k of ['neutral', 'elevated']) {
  const { delays, ...s } = res[k].summary;
  out[k] = s;
}
console.log(JSON.stringify(out));
