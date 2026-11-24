# Proposed method: Macro + Micro

The Phase28–29 source is deliberately **not relocated**, preserving the
existing training and evaluation runs. This folder provides explicit
entrypoints for new experiments and the code map:

- src/game/nine_ball_mcts_agent.py — proposed hierarchical agent
- src/game/nine_ball_mcts_v3.py — search with candidate priors and State V
- src/game/nine_ball_learned_agent.py — goal-conditioned Micro SAC
- src/game/nine_ball_policy_v3.py and nine_ball_value_v3.py — Macro models
- src/game/nine_ball_encoding_v3.py — observation and tactical-goal encoding
- scripts/iterate_nine_ball_hierarchical_v3.py — self-play and learning
- scripts/eval_nine_ball_hierarchical_v3.py — proposed model evaluation

Use src.game.agents.proposed.hierarchical in new experiments.
Compared agents are separate under src/game/agents/baselines.
Physics, rules, and encoding belong to shared game/env modules.
