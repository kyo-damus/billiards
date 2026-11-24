# Chen & Li (2019) nine-ball Macro–Micro RL: source-to-code plan

**Verified source:** Yu Chen and Yujun Li, *Macro and Micro
Reinforcement Learning for Playing Nine-ball Pool*, IEEE CoG 2019,
https://ieee-cog.org/2019/papers/paper_162.pdf
DOI: https://doi.org/10.1109/CIG.2019.8848113

## Core method (retain these distinctions)

1. Macro action A is a shot aiming path: target ball through a choice
   of pocket and/or cushions (represented by a baseline angle psi).
2. Micro action is five-dimensional: delta-psi, cue speed, elevation,
   horizontal cue-tip offset, vertical cue-tip offset.
3. An actor pi(s,A) proposes a micro action. A learned candidate Q(s,A)
   estimates Macro quality. Both share a CNN over ten 62×125 feature
   maps with positional, aiming-path and table geometry channels.
4. Search contains common physical states and aiming-path sub-nodes.
   UCB selects Macro candidates; Gaussian perturbations of the Micro
   actor are sampled using a renewal/restart exploring criterion.
5. The selected Macro maximizes estimated Q. The chosen Micro action
   is the explored shot with best next-state Q.
6. Training combines squared TD loss for Macro Q and L2 regression
   to the selected searched Micro action. This is *not* equivalent to
   the current project's standalone one-shot SAC or visit imitation.
7. Paper's training/search specifications: gamma=0.95,
   exploration alpha=0.95, max renewal steps 200,
   Adam lr=1e-4, batch size 32; distributed resources differ greatly.

## Required nine-ball adaptation

- This paper already studies nine-ball, so the legal first contact,
  1–9 inventory and lowest-ball ordering can be kept.
- Replace paper's original physics/state renderer with the common
  FastFiz/RackState and WPA-style rack/match engine.
- Add ball-in-hand, illegal break and push-out handling in a clearly
  isolated rules adapter; do not silently pass these to the actor.
- Preserve distinct aiming-path (Macro) proposals, including at least
  explicitly described direct-pocket vs cushion-path coverage.
- Preserve the actor-imitation-of-best-search-micro learning target
  and its relation to Q. Do *not* label a shared-SAC/MCTS variant
  as a reproduction of this method.
- Convert rendered 10×62×125 maps consistently from the same
  physical board; verify physical coordinates and ball-index mapping.
- Compare with original paper's *run-out* metric as a secondary
  special-purpose test, alongside the thesis's race-to-9 paired matches.
  The reported 40.6% was on the paper's simulation and should never
  be compared as if directly measured under WPA rules.

## Implementation stages

- Stage 0 (completed): verify source, design mapping and constraints.
- Stage 1: independent goal/path renderer, legal shot options and
  fixed Micro output decoder with property-based geometry tests.
- Stage 2: convolutional actor and candidate-Q networks, checkpoint
  versioning and finite forward/backward smoke.
- Stage 3: renewal-based Micro search within Macro UCB, with test
  records distinguishing physics failure from a game foul.
- Stage 4: independent training pipeline for Q TD targets and Micro
  imitation targets (never load Phase29 learned Macro weights).
- Stage 5: small DGX smoke, controlled training and paired evaluation,
  then publish compute budgets and major fidelity deviations.

Current status: source mapping ONLY. There is no Chen–Li reproduction
model or training pipeline in this branch. Do not cite it as implemented.
