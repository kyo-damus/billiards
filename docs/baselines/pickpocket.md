# PickPocket-inspired 9-ball Monte Carlo baseline (v0)

## Source documents (verified)
- Smith (2006), *Running the Table: An AI for Computer Billiards*,
  AAAI, https://webdocs.cs.ualberta.ca/~jonathan/PREVIOUS/Papers/Papers/pickpocket.pdf
- Smith (2007), *PickPocket: A computer billiards shark*, Artificial
  Intelligence 171(16–17):1069–1091,
  https://doi.org/10.1016/j.artint.2007.04.011

This is a **modified 9-ball adaptation**, NOT a faithful reproduction,
not a measured reproduction of Smith's original tournament results,
and not evidence of strength until the smoke/held-out evaluations run.

## Mechanism and mapping

| Source mechanism | Current implementation | Changed / omitted |
| --- | --- | --- |
| Continuous five-parameter physical shots | Independent physical shot templates in 5D (angle, speed, elevation, tip x/y), same FastFiz simulator | On-the-fly limited direct-pocket candidates; no precomputed minimum-power table |
| Vary speed and spin for position | Several speeds and vertical-tip offsets | Limited samples; no optimization over all horizontal tip offsets |
| Stochastic execution error | Independent zero-mean Gaussian perturbation of the five *physical parameters* inside planning | Custom standard deviations: 0.25 deg azimuth, 0.05 m/s speed, 0.3 deg elevation, 0.35 mm each tip offset. **NOT Smith's experimentally chosen noise parameters** |
| Monte-Carlo evaluation of outcomes | Average one-step physical outcome scores per proposal | Root-only search, not original 2-ply MC; omitted success-probability lookup table |
| Leaf state quality | Legal exact-pocket pot, shooter follow-up direct options, opponent direct options, fouls/win | Simplified 9-ball heuristic, not original top-three weighted probability lookup |
| Safety under low success probability | Root-stage sampled short/small-angle safety candidates if max pot rate < 0.65 | Crude subset, not original broad safety search |
| Bank/kick/combinations | Not implemented | **Major fidelity gap** |
| Eight-ball legal object choice | Smallest-numbered object ball is required by 9-ball rules | Major necessary rule adaptation |
| Break shot | Same frozen learned BREAK Micro actor as proposal | Shared component for control; ordinary shots do **not** use proposed Micro |
| Execution uncertainty | Noise is added during decision-time simulations | Actual match execution remains deterministic FastFiz; future robustness test should add identical execution noise to *both* sides |

## Fairness / interpretation

- Break policy, race-to-9, WPA break/illegal-break rules, FastFiz, seed
  pairs, and held-out rack layouts match the Phase29 proposal evaluator.
- PickPocket v0 is not a No-Macro ablation: it is independent symbolic
  candidate generation and stochastic physical search.
- The default search budget is 8 nominal candidates × 3 noisy samples
  plus nominal validity tests and up to 6 safety probes. This differs
  substantially from Phase29's 24-simulation/depth-2 MCTS. Always
  report native physics calls and wall-clock time before claiming a
  compute-normalized comparison.
- Cases where no legal physical shot can be executed remain unfinished
  or are separately diagnosed, not counted as a loss by fiat.
- Do not call the results "PickPocket vs proposed method" without
  the "PickPocket-inspired v0" qualifier.

## Next fidelity milestones

1. Record root candidate evaluation traces, physical-call counts,
   safety selection, exact-pocket pot rates, native failures.
2. Use held-out physical-shot calibration to estimate execution noise
   (or report a sensitivity sweep). Avoid noise calibrated on test wins.
3. Add actual 2-ply Monte-Carlo continuation and the original
   mobility-value idea with per-shot pot probability approximation.
4. Add bank/kick/combinations only if feasible, and document coverage.
5. Only after those milestones choose whether to call it a
   PickPocket *adaptation* strong enough for the main comparison table.

Executable module:
- src/game/agents/baselines/pickpocket/agent.py
- scripts/comparisons/eval_nine_ball_baselines.py --baseline pickpocket_mc
