# Rule/environment adaptations for nine-ball research baselines

**Scope:** Explain which parts of prior billiards agents are *necessarily*
changed to operate under the project's WPA-inspired nine-ball rules and
FastFiz simulator; identify methodological compromises separately.
This is an implementation/research-planning document, not a claim that all
reference methods have been reproduced.

## References

1. M. Smith, "PickPocket: A computer billiards shark," *Artificial
   Intelligence* 171(16-17), 2007, doi:10.1016/j.artint.2007.04.011.
   Earlier algorithmic detail: M. Smith, "Running the Table: An AI for
   Computer Billiards," AAAI 2006:
   https://webdocs.cs.ualberta.ca/~jonathan/PREVIOUS/Papers/Papers/pickpocket.pdf
2. C. Archibald, A. Altman, Y. Shoham, "Analysis of a Winning
   Computational Billiards Player," IJCAI 2009:
   https://ijcai.org/proceedings/2009/231
3. Y. Chen, Y. Li, "Macro and Micro Reinforcement Learning for Playing
   Nine-ball Pool," IEEE CoG 2019:
   https://ieee-cog.org/2019/papers/paper_162.pdf

## Adaptation vs reproduction: a three-way classification

- **Required by target game/environment:** change legal-shot selection,
  rule and terminal-state evaluation, physics I/O adapters, and
  environment-specific score calculations.
- **Chosen for fair evaluation:** same official-style rack/match setup,
  seed and layout split, physics, wall-clock/physics-call budgets, and
  initial player conditions. Such decisions are not logically required
  by an algorithm and must be disclosed.
- **Algorithmic deviation:** replacing the paper's candidate generator,
  leaf evaluation, lookahead, network, sampling distribution, learning
  target, or trained micro actor. Report this separately as incomplete
  reproduction or inspired adaptation, never silently as the paper's
  method.

## Matrix

| Comparison | Original setup | Required change for target nine-ball | Core method to preserve | Optional/fidelity-sensitive deviations |
| --- | --- | --- | --- | --- |
| PickPocket (Smith 2007) | Simulated tournament eight-ball. Choices depend on the shooter's solids/stripes and subsequent eight-ball clearance. Five real-valued cue parameters; probabilistic simulation and search | Replace eight-ball group restrictions with lowest-ball-first contact, legal combos and nine-ball game completion; replace own/opponent available-pot counts and leaf mobility evaluation accordingly; remap pocket/physical shot parameter conventions | Difficulty/success probability estimates, physical error sampling, bank/kick/combination coverage where reported, Monte Carlo/probabilistic search, position/safety evaluation | Present root-only PickPocket-inspired v0 omits full two-ply search, probability tables, broad collision classes, calibrated noise; shares BREAK-only frozen Micro; match execution currently deterministic |
| CueCard (Archibald et al. 2009) | Simulated eight-ball, sampled-shot clustering, two-level lookahead; late-game noiseless run-out search and opponent Markov model indexed by own/other group's remaining ball counts; optimized eight-ball break | Adapt legal candidates to ordered nine-ball target; redefine game-over and post-shot ownership; *replace/retrain* eight-ball-specific opponent win-probability Markov model with nine-ball state, or explicitly replace with another estimator; re-optimize break criteria and foul handling for WPA-style nine-ball | Shot sampling, outcome-state clustering and weighted representative states, two-level and late-game search distinctions | A simple two-ply Monte Carlo without clustering/late-game Markov is not CueCard reproduction; use of a shared break is evaluation choice, not the original method |
| Chen & Li (2019) | Nine-ball on separate physical simulator; Macro = aiming paths (pocket and cushions); Micro = five continuous controls; Q(s,A) + pi(s,A), convolutional feature maps, UCB/renewal exploration, TD Q and imitation of best searched Micro; paper reports run-out performance | Keep ball ordering, but map to current physics/shot encoding and WPA-style foul/break/push-out/ball-in-hand transitions; add match/rack context and consistent terminal/reward interpretation if evaluating/training for race-to-N; regenerate features in current coordinates | Aiming-path Macro proposals including cushion paths, CNN representation if reproducing, Q/actor plus search guidance, Q TD target and searched Micro imitation target | Swapping in project's SAC, MCTS Policy/State V, or match-outcome loss would be a new hybrid, not faithful replication. Original run-out rate must not be compared directly to race-to-N win-rate |
| Pot-only Greedy (project-defined) | No original paper; adapted internal ablation | Define legal lowest-ball and target-pocket choice, WPA engine handling | Pot-only deterministic Macro rule; shared 5D Micro | Not a published baseline: label as pot-only geometric heuristic |
| Uniform MCTS (project-defined) | No original paper; adapted internal ablation | Reuse current candidates and WPA-style outcome engine | Macro candidate generation and MCTS without learned Policy and V; shared 5D Micro | Not a no-Macro model, not PickPocket, and not Chen–Li's pure tree-search implementation |

## Detailed unavoidable semantic changes for eight-ball baselines

### Target choice and legal first contact

Eight-ball: a player often has several legal object balls in their
assigned group; the eight-ball becomes a winning target after group
clearance. Nine-ball: a legal first contact generally targets the
lowest-numbered remaining ball, even where a *later* collision pots
the nine-ball. The set of candidates and any success-probability
estimator over candidate classes must reflect this difference.

This is not equivalent to always requiring the lowest ball to be the
pocketed ball: such a rule would disallow legal combination shots in
nine-ball. Map first contact, pocketed balls, opponent turn, fouls,
and actual nine-ball win condition separately.

### Position-play value and opponent mobility

Eight-ball positional planning can consider multiple same-group balls
as immediate legal continuations. Nine-ball normally has one
first-contact target. A leaf state's "number of available shots",
"probability of running out", and opponent's threat must be
redefined for the lowest legal ball, break/foul status, and next rack.
Preserve the *form* of the paper's evaluation whenever possible,
but replace rule-specific features and retrain/calibrate estimates.

### Terminal states, break, and turn transitions

Original eight-ball rack outcomes cannot be reused verbatim.
Target match uses race-to-N, alternate break, WPA-inspired break
legality, push-out/table-choice branches, and fouls/ball-in-hand.
For published nine-ball models, the *game* still matches, but these
match-level and tournament-specific transitions are extra. The
project's own nine-ball rules should not be called a complete
implementation of every WPA tournament regulation without verification.

### Physics and uncertainty

Physics simulator replacement is a *common-platform choice*, not
a theorem-required change to the algorithm. Physical units and
parameters must be converted, and model calibration may be necessary.
When stochastic search is integral to the paper, a decision-time
noise model is not the same as stochastic **execution**. Report
noise parameters and ideally test identical execution errors for
all compared methods. Using deterministic execution may make
uncertainty-robust strategies look artificially weak or strong.

### Shot-budget normalization and break ablations

An equal number of MCTS simulations is not automatically an
equal number of FastFiz calls (candidate generation, sampling,
and leaf rollouts also cost physics). Record physics calls, decision
time, hardware, success, win rate, and unfinished runs.

CueCard explicitly showed break-shot choice mattered; when using
a common frozen break actor, label results "shared-break
controlled comparison". Preserve each method's own break in a
secondary method-faithful comparison if practical.

## Interpretation guidance

For the thesis, distinguish:

1. **Common-environment adapted method performance:**
   strength under our nine-ball rules, with all adaptations listed.
2. **Algorithmic ablations:** shared Micro and/or break to isolate
   Macro components. These cannot be presented as an independent,
   full-paper reproduction.
3. **Paper-faithful reproduction:** only when the source's key
   decision, evaluation, exploration, and learning mechanisms
   are retained to an explicitly justified fidelity level.
4. **Cross-game transfer:** nine-ball pretrained weights adapted
   to the original eight-ball environment require additional
   experiments. A model *retrained from scratch* in eight-ball
   demonstrates trainability/portability, not zero-shot
   generalization. Measure frozen-weight transfer and low-shot
   fine-tuning separately.

## Thesis-ready explanation

> The selected reference methods were designed for different game
> rules and physics engines. To compare the *adapted* algorithms on
> equal conditions, we unified the simulator, nine-ball rules,
> match format, starting layouts, and evaluation budget. Rule-specific
> candidate legality, state evaluation, terminal conditions, and
> break/foul processing were adapted for nine-ball. The reference
> method's distinctive search, uncertainty handling, and learning
> mechanisms were retained wherever feasible. Any departures from
> these mechanisms are itemized as implementation limitations;
> therefore, the experiments evaluate **nine-ball adaptations of
> the cited methods**, not unmodified reproductions of the original
> systems or their reported tournament win rates.
