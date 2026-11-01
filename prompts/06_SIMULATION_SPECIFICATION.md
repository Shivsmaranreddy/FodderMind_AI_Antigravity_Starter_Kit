# Prompt 06 — Simulation specification

Design a small, reproducible simulation only for decisions the project can defensibly represent, such as price changes, feed availability, inventory, procurement, budget, and planned feed allocation.

Before coding, document:
- state, actions, transitions, episode definition, decision time step, termination conditions;
- which elements are evidence-based versus assumptions;
- random variables and distributions, with provenance or explicit assumption labels;
- metrics and reward candidates;
- what the simulator does NOT model, especially animal biology unless validated.

Choose a decision-relevant time step (e.g. daily feeding or weekly procurement); do not create per-second data merely to increase row count. Separate simulator code from any biological model. Ask for approval of the specification before implementation.
