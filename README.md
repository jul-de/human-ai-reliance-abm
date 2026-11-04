# Agent-Based Model: Reliance Recalibration Under Silent AI Degradation

*Julia Donovan · built in Python (NumPy/pandas) for an LSE MSc Behavioural Science module on agent-based modelling.*

## The question

Human oversight is the main safeguard when AI systems are deployed in high-stakes settings, but it only works if people actually notice when an AI starts failing and change how much they rely on it. This model asks: **when an AI assistant silently becomes less reliable, do users recalibrate their reliance, how fast, and what makes recalibration fail?** It focuses in particular on whether *anthropomorphic framing* (presenting the AI as warm/human-like) delays recalibration.

## The architecture (and why it's interesting)

The model separates two things that usually get collapsed together, borrowing the logic of **Signal Detection Theory**:

- **Sensitivity (α):** how quickly an agent *registers* that the AI is making errors (a delta-rule / Rescorla–Wagner-style update on a reliance value).
- **Decision criterion (θ):** how much accumulated evidence an agent needs before it *acts* — i.e. switches away from relying on the AI.

The key theoretical move: **anthropomorphism is modelled as a criterion (θ) shift, not a sensitivity (α) shift.** Anthropomorphic agents detect degradation just as well, but require more accumulated evidence before they change their behaviour. This is anchored in Peters & Scharlau (2025), who found that detection and action dissociate — people can register that an AI has degraded without changing how they use it. The model makes that dissociation explicit and testable.

## What the model shows (four versions)

- **v1 — baseline:** a delta-rule reliance update with a fixed switching threshold reproduces the expected rise-then-fall pattern (reliance builds pre-drift, falls after the AI degrades). Confirms the mechanism works before any manipulation.
- **v2 — the core prediction:** holding sensitivity (α) identical, shifting only the criterion (θ: 0.25 → 0.40) for anthropomorphic agents produces a 16-trial delay in switching and a ~2.7× increase in cumulative cost (trials spent still relying on the degraded AI). The effect is driven entirely by criterion, not detection.
- **v3 — robustness + emergence:** drawing α and θ from distributions (a heterogeneous population of agents) and pooling across runs shows the effect is robust, not an artefact of specific parameter values. **The most interesting result is emergent and unprogrammed:** ~8.4% of anthropomorphic agents *never switch* within the window (vs ~0.4% of neutral agents). This never-switch rate arose from the interaction of heterogeneous criteria with a stochastic AI output stream — it can't be read off the parameters analytically, which is exactly what justifies an agent-based approach for this question.
- **v4 — ecological validity:** replacing abrupt degradation with gradual drift (closer to how real AI degrades) *amplifies* the problem — never-switch rates climb substantially under gradual drift. Recalibration failure is worse, not better, when degradation is slow and silent.

## How to read the validation honestly

This is a **middle-range, conceptual model** doing **generative (pattern-oriented) validation** — it asks whether a principled mechanism reproduces known qualitative patterns from the trust/automation-bias literature (asymmetric updating, prior-anchoring, dose-response to error frequency). **It is not calibrated to a specific dataset, and it is not validated against individual human behavioural data.** Its contribution is mechanistic: it shows that separating *detection* (sensitivity) from *action* (criterion) generates specific, differential predictions — e.g. that interventions targeting criterion-crossing should outperform those targeting detection — which can then be tested empirically.

## Files

- `PB440_Model_v1.ipynb` — the model (code + inline markdown explaining each version).
- `PB440_Model_v1.html` — rendered version with all outputs and plots visible (readable without running).

## A note on AI assistance

I used AI tools (Claude) to help translate the architecture into code and for debugging — the theoretical design (the SDT-inspired sensitivity/criterion separation, the anthropomorphism-as-criterion-shift choice, the scope and predictions) and the interpretation are mine, and I can justify every parameter and modelling choice.
