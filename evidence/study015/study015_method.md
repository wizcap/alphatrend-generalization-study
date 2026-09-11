# Study015: fixed generalization comparisons for AlphaTrend exits

Declared2026-09-11 before fitting/evaluating this study. All source history has already been used in research; none is a fresh holdout. Preserve studies001–014, the shadow controller and its state. This experiment changes research exit learners only; no live orders, scheduler, production selection, new symbols or post-result tuning.

## Hypotheses and fixed interventions

Keep BTCUSDT/ETHUSDT,2h/4h, the same355 pinned15m market/mark/funding archives, exact study012 natural-exit labels, monthly dates January2024–August2026, six half-year accounting windows and study013 entry-latched model timing. Each model can only close once at24h after an AlphaTrend entry decision. Natural exits take priority; initial synthetic window episodes retain Alpha; no entry veto, extra entries or depth features.

Seven pooled arms: AlphaTrend; original18-feature monthly HGB; compact8; robust8; temporal_mean; temporal_guard; ridge8. Three additional transfer arms: transfer_current, transfer_compact, transfer_robust. In each transfer arm BTC is managed by a model trained only on ETH and ETH only on BTC. This tests transfer between two already-known assets, not generalization to an unseen market universe.

Fixed8 features, chosen by economic role rather than fitted importance: alpha_distance, alpha_slope, rsi14, atr14_relative, taker_fraction, risk_vol, pnl_atr, giveback_atr. The constant entry-state column, several overlapping trend/volume indicators and redundant path summaries are omitted. This is a package comparison, not attribution to one feature or one hyperparameter.

- Original current learner: exact study013 expanding models,18 inputs,100 boosting iterations,7 leaves,min20 per leaf,L2=10,learning rate.05,seed11. Reuse64 immutable saved versions and verify their predictions/curves. Single-source current transfer heads use these same settings.
- compact8:8 inputs,100 boosting iterations,3 leaves,max_depth2,min20 per leaf,L2=50,learning rate.03,seed11,no early stopping. Raw economic labels, unit sample weights, expanding history.
- robust8: same compact model; winsorize training labels at that cohort's unweighted2.5%/97.5% quantiles (linear interpolation). Retain all rows. Assign weights proportional to1/sqrt(n_group) for symbol × decision-half-year groups, normalized to mean1. This moderates density imbalance without claiming equal group mass. Both scaler and learner receive the weights. Preprocessing is fitted using only that cohort's matured training observations.
- temporal_mean: arithmetic mean of three robust8 heads trained on expanding history, trailing730 days and trailing365 days of checkpoint decision times before the cutoff. Fixed equal weights. Overlap and identical early cohorts are explicitly reported; members are not independent samples.
- temporal_guard: same three heads; early exit only if all three predictions are strictly negative (equivalently their maximum<0). Disagreement means retain AlphaTrend. This is conservative agreement, not a calibrated uncertainty/confidence interval.
- ridge8: same8 inputs, clipped labels and group weights as robust8, with fixed Ridge(alpha=50,solver=cholesky) and training-weighted StandardScaler. It is a separate simple-model control.

No automated parameter search. Architecture/feature restriction and robust preprocessing each combine several fixed choices, so their effects are not individually identified. Winsorizing changes the estimated target away from raw expected payoff; evaluation always uses the actual unclipped economic outcomes and costs.

## Causality and sample accounting

At each month, consume only non-left-censored exit labels with natural payoff known strictly before month minus7 days. Window-limited cohorts additionally require checkpoint decision >= cutoff minus the declared day count. Transfer cohorts remove the target asset before all preprocessing, including target quantiles, weights and scaling. No current target-asset statistics, labels, rankings or future-survival filters enter training. Row identity/schema/timestamps are validated; future rows do not influence fitted state. Fewer than20 matured training rows produces an explicit no-model abstention (Alpha retained), not discarded dates. An ensemble with any insufficient member abstains as a whole. Report all such slots and action counts.

Per month/timeframe: five new pooled component heads (compact/all,robust/all,robust/730,robust/365,ridge/all), plus six transfer heads (three families × two held-out assets).32 ×2 ×11 =704 new fit slots, plus64 cached original references. Not every slot is necessarily trainable; report exact fitted/abstention counts. Predictors are pinned at natural entry through the checkpoint even if a newer month arrives. All models are hypothetical historical versions, not timely live predictions.

## Evaluation and inference

All10 policies per timeframe,6 windows,3 scenarios and2 assets:720 asset-accounting windows (not independent statistical samples) /360 portfolio windows. Publish scalar accounts with15m execution delay,7bps per side and actual settled mark-price funding; stress14bps at15m and7bps at30m. Month updates do not flatten positions; the same half-year reporting boundaries reset/terminally close, matching study013. No control/gate selection changes after results. The adaptive controller itself is not being retuned or updated.

Primary development period2024–25; known2026 throughAugust31 20:00UTC is temporal context. Show annual/per-symbol return, drawdown, orders, fees/funding, full equity, loss repair and winner retention. Matched baseline and current models must reproduce study013 for every scenario and period. Report transfer for both asset directions, including any lack of training support.

Predeclared paired contrasts per timeframe: compact8 vs current; robust8 vs compact8; temporal_mean vs robust8; temporal_guard vs temporal_mean; ridge8 vs robust8; transfer_current vs current; transfer_compact vs compact8; transfer_robust vs robust8; temporal_guard vs current; and temporal_guard vs Alpha.20 contrasts,40 adjusted intervals. Paired circular blocks7/28 days within the four development half-years,100000 draws,seed1015,tail.025/40. These do not correct for all prior studies and do not validate a selected winning configuration. Recent2026 results are not used to tune/select any arm.

Diagnostic prequential predictions are retained for every actual checkpoint action, including censored/boundary/abstaining cases. For payoff-error/regret summaries use only matched, naturally resolved non-left-censored baseline labels; explicitly report unmatched/censored counts. Score against original unclipped y. Correct-action sign is not an economic profitability metric. No diagnostics feed back into fit or policy selection.

## Validation and review

Before fitting, test strict label maturity, window inclusion, held-asset exclusion, future-label/held-asset mutation invariance, all preprocessing arithmetic, model-shape constraints, insufficient cohorts, conservative agreement and monthly version latching. Verify source/evidence hashes and freeze method,code,tests,runtime before real fits. Independently reconstruct training membership/quantiles/weights/scalers, refit every new trained head, manually traverse trees or solve weighted Ridge, replay all targets/actions and scalar-vs-compiled accounts, recompute portfolio/asset/repair summaries and40 intervals. All saved original monthly references must match prior evidence. Reconcile full curves, not only final returns.

Request Claude focused design/code and result reviews using explicit supplied text with tools disabled; retain responses and corrections. If unavailable, disclose failed attempts and keep executable independent validation mandatory. No live/shadow-controller deployment is authorized by a positive backtest; production_selection remains null.

References: [HGB regularization parameters](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html), [Ridge regularization](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [training-only preprocessing](https://scikit-learn.org/stable/common_pitfalls.html).

Pre-fit Claude review corrections: payoff diagnostics use the component mean, never the guard maximum; insufficient versions have no payoff forecast but their hold-action regret remains included when resolved. Raw-y prediction error compares different fitted estimands and is descriptive only, not a model-quality ranking. Report identical temporal cohorts, non-degenerate months and constant-tree fits. Known-asset transfer also inherits prior feature/design selection on both assets. Decision collisions fail explicitly; natural exits already take priority in the shared signal generator.

