# Study015 review resolution

Claude's original design input and response are preserved. These are review opinions, not executable instructions.

Before fitting, accepted and addressed: explicit collision assertion; accounting-window counts are not independent observations; insufficient cohorts have no payoff forecast; temporal_guard stores a policy maximum but diagnostic payoff prediction uses the component mean; raw-payoff MAE is descriptive because winsorization changes the target. Added tests for abstention and score semantics. Reports enumerate constant learners and temporal cohort overlap.

Context corrections: the frozen signals() generator already prioritizes natural exits and produces unique decisions; its checkpoint and entry cannot collide in the stated scenario. The added assertion is defensive. All64 cached expanding references were previously actually fitted and are frozen by their original byte hashes. Early temporal cohorts are compared by exact training IDs, not by an assumed calendar cutoff; Claude's claim that all three coincide until2025 is not used. Correlated ensemble votes are never described as a confidence interval. Regret scores the actual retain/exit action against original payoffs, independent of whether a forecast targets clipped or raw y.

The new research controller does not exist: study015 only defines hypothetical research learners and policies. Study014 files and state remain pinned. No production candidate is selected.

Result review and disposition will be appended after independent verification.


Implementation review (completed while fixed evaluation ran):

- Rejected the claim that regret assumes zero early-exit P&L. The exact frozen exit label in repair_v12.labelled_events is natural_return minus early_return (line211), not total natural-trade return. Thus max(y,0) on exit and max(-y,0) on hold are the two correct action regrets. The interpretation remains diagnostic, not additive portfolio money.
- The 0. positional simulate argument is slippage_bps, not funding. The7bps scenario is total one-way modeled trading friction. Historical funding is read separately from market funding and mark arrays and reconciled by the compiled account audit.
- Both asset readers already require their entire index to equal the same explicit15-minute UTC grid (sources_v8.load). The first-symbol reporting clock is therefore shared, not inferred merely from array length.
- Alpha is first in the frozen ARMS list. No previous-symbol baseline can be silently substituted: baselines are keyed by symbol and recreated for each scenario. Missing Alpha would raise; the fixed ordering is unchanged.
- The64 cached pickles are immutable files already independently verified in study013. They contain matching names and no fitted flag; the704 new-fit count is checked against the actual new-model registry by verify_v15. Original complete curves and report fields must match study013.
- Rebalance band0.05 is unchanged from study013 and is explicit in the frozen code. The method declares the same accounting protocol by reference; this note makes the parameter explicit without changing the pre-fit method.
-40 intervals cover20 contrasts under two block analyses. Bonferroni across all40 is intentionally conservative. Each extreme bootstrap tail has only about62 draws below it, so endpoints have Monte Carlo uncertainty. These intervals are not a global correction for earlier research.

No strategy, hyperparameter, label, cost, source, test or frozen implementation was changed in response to the implementation review. Result interpretation will explicitly retain the limitations above.


Results review completed against the complete canonical results while the independent full-account audit was still running. Claude confirmed the annual-return arithmetic, all20 log-wealth differences, slot counts, interval sign counts and regret algebra. The original brief explicitly discloses that audit timing; its text is preserved.

Accepted interpretation limits:2026 comparisons are descriptive without2026 intervals; one negative28-day interval does not establish harm across both block choices; multiple comparisons and Monte Carlo tails limit confidence; point-ranking by return alone is not a model selection rule. Few early exits/constant early heads cannot establish transferred skill. The report discloses action counts and both execution stresses.

Drawdown clarification: grouped() concatenates returns and computes a new cumulative curve before its running-peak drawdown. It is not max(separate half-year drawdowns), as the review suggested. It is nevertheless a synthetic account under the fixed half-year reset convention, not the continuous study014 account. Delivery independently recomputes each complete curve's running-peak drawdown.

Investigated matching rounded fields:4h transfer_compact and transfer_robust have the same development loss repair because only one development episode differs materially between them:2025H2 BTC episode7, a baseline WINNER. Its candidate log contribution changes by0.0325761373880915; the winner-retention difference is expected while losing-episode repair remains equal to floating-point tolerance. There is no copied column.4h transfer_current has12 actual early exits in2026, with return-0.04963863321681228 vs Alpha-0.0496421205719394; the3-decimal match is rounding, not a duplicate strategy.

Remaining source-context questions resolved: frozen y includes both arms' modeled fees and actual funding; bootstrap starts and offsets are drawn separately within each half-year, never across reset boundaries; path features read entry execution through the bar BEFORE the checkpoint (reference_state and exit_features), not later episode highs/lows. Independent executable checks, not the review response, determine final verification status.


Final executable verification completed successfully: 303 tests, 704 independent refits, 720 complete asset curves, 23010 actions, 12210 attributions, 3600 diagnostic rows and 40 intervals. All original Alpha/monthly curves reproduce study013. All frozen source/evidence hashes and study014 shadow state are unchanged. No deployment selection.
