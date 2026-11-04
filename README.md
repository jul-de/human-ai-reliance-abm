# Trust Recalibration Under Silent AI Degradation: An Agent-Based Simulation

*Julia Donovan · built in Python (NumPy) for an LSE MSc Behavioural Science module on agent-based modelling.*

A computational model exploring how individuals update trust and change reliance when an AI system's reliability deteriorates without explicit notification. The model introduces heterogeneous learning rates and behavioural switching thresholds, then examines how different degradation trajectories affect recalibration. It explores a theoretical distinction between learning from observed AI outcomes and deciding when to change reliance.

![Switching delay and exposure to degraded AI across drift trajectory and threshold](figures/trust_recalibration_v4.png)

## The question

Human oversight is the main safeguard when AI systems are deployed in high-stakes settings, but it only works if people actually notice when an AI starts failing and change how much they rely on it. This model asks: **when an AI assistant silently becomes less reliable, do users recalibrate their reliance, how fast, and what makes recalibration fail?** It focuses in particular on the hypothesis that *anthropomorphic framing* (presenting the AI as warm or human-like) delays recalibration.

## The architecture

Each agent updates its trust in the AI after every outcome with a delta rule, T<sub>t+1</sub> = T<sub>t</sub> + α(O<sub>t</sub> − T<sub>t</sub>), and stops relying on the AI once its trust has fallen more than θ below its starting level. The model keeps two things separate that usually get collapsed together:

- **Learning rate (α):** how quickly trust updates in response to AI outcomes.
- **Switching threshold (θ):** how much accumulated trust loss is needed before the agent changes its reliance.

The separation is **inspired by Signal Detection Theory**'s split between sensitivity and decision criterion. It is an analogy, not a formal implementation: α and θ are a learning rate and a behavioural threshold, not estimates of d′ or c.

The key theoretical move: **the social-framing hypothesis is modelled as a higher switching threshold, not a lower learning rate.** Agents in the elevated-threshold condition register errors just as quickly, but need more accumulated trust loss before they change behaviour. This is anchored in Peters & Scharlau (2025), who found that detection and action dissociate: people can register that an AI has degraded without changing how they use it. The model does not simulate an anthropomorphic interface directly. The elevated threshold stands in for a proposed mechanism through which social framing could affect reliance.

## Outcome measures

The switching rule is active from the first trial, so a run of errors can make an agent switch while the AI is still reliable. These pre-drift switches are reported as **false alarms**. For agents still relying on the AI when it degrades, the model reports three outcomes together:

1. the proportion who switch within the simulation window
2. the mean switching delay among those who switch
3. the mean exposure to the degraded AI (post-drift trials of continued reliance) across all of them

Exposure is censored at the end of the window, so for agents who never switch it is a lower bound.

## What the model shows (four versions)

- **v1, baseline:** a delta-rule update with a fixed threshold reproduces the expected rise-then-fall pattern. 7 of 50 agents false-alarm before the drift; all 43 remaining agents switch, a mean of 8.7 trials after it. This confirms the mechanism, and that the threshold trades off false alarms against slow detection of real degradation.
- **v2, the core prediction:** with α held identical, raising θ from 0.25 to 0.40 increases the mean switching delay from 8.7 to 19.3 trials and mean exposure from 9.7 to 21.5 trials (about 2.2×). It also removes false alarms (4/25 → 0/25). The difference is driven entirely by the threshold.
- **v3, robustness and emergence:** drawing α and θ from distributions and pooling 10 runs shows the effect is robust (delay 10.2 vs 16.9 trials; exposure 11.4 vs 20.6). **The most interesting result is emergent and unprogrammed:** 8.4% of at-risk elevated-threshold agents *never switch* within the window, vs 0.5% of neutral agents. This arises from the interaction of heterogeneous thresholds with a stochastic AI output stream. It can't be read off the parameters analytically, which is what justifies an agent-based approach. The trade-off shows up at population level too: 12.8% of neutral agents false-alarm, vs 0.4% of elevated-threshold agents.
- **v4, drift trajectory:** replacing abrupt degradation with gradual drift (closer to how real AI degrades) *amplifies* the problem. The never-switch rate in the elevated-threshold condition rises from 7.7% to 19.2%, and mean exposure from 21.7 to 33.2 trials. Recalibration failure is worse, not better, when degradation is slow and silent.

## How to read the validation honestly

This is a **middle-range, conceptual model** doing **generative (pattern-oriented) validation**. It asks whether a principled mechanism reproduces known qualitative patterns from the trust and automation-bias literature (asymmetric updating, prior-anchoring, dose-response to error frequency). **It is not calibrated to a specific dataset, and it is not validated against individual human behavioural data.** Nor does it model the consequences of reliance: exposure counts trials of continued reliance, not harm. Its contribution is mechanistic. It shows that separating *updating* (learning rate) from *acting* (switching threshold) generates specific, differential predictions that can then be tested empirically, for example that interventions targeting threshold-crossing should outperform those targeting detection.

## Files

- `PB440_Model_v1.ipynb`: the model (code plus inline markdown explaining each version).
- `PB440_Model_v1.html`: rendered version with all outputs and plots visible (readable without running).
- `figures/`: the v2–v4 figures.

The version originally submitted for the module is tagged [`v1-submitted`](https://github.com/jul-de/human-ai-reliance-abm/tree/v1-submitted). The current version separates pre-drift false alarms from post-drift recalibration, reports switching delay and exposure for all at-risk agents, and reframes α/θ and the anthropomorphism condition more precisely.

## A note on AI assistance

I used AI tools (Claude) to help translate the architecture into code and for debugging. The theoretical design (the SDT-inspired separation of learning rate and switching threshold, the threshold-shift operationalisation of social framing, the scope and predictions) and the interpretation are mine, and I can justify every parameter and modelling choice. The revision after submission (separating false alarms, the revised outcome measures and the reframing) was also made with Claude's help, following an external review.
