# Manuscript review and disposition

The preprint received an automated Claude editorial critique after the original frozen study had completed. This is an AI-assisted internal check, not external scientific peer review. The original study's separate review record is retained in `evidence/study015/study015_review_resolution.md`.

## Clarifications incorporated

- Distinguish left-censored episodes from unresolved right-censored labels. Both are excluded from training; boundary protection is a separate policy rule.
- Make clear that clipping quantiles are computed on the empirical training-row distribution, while group weighting applies to scaling and fitting. The robust package changes the estimand and is not a one-factor ablation.
- Report small-cohort constant boosting models. A valid constant prediction can trigger an early exit; it is not automatically a no-op. No pooled new HGB head was constant in this study.
- Label the highlighted Ridge-versus-Current and temporal-mean-versus-Current return comparisons as exploratory summaries rather than additional declared inference tests.
- Include every declared interval, distinguish interval-level and contrast-level counts, and explain overlapping cohorts, circular joins, extreme-tail Monte Carlo uncertainty, and the absence of a prospective power target.
- Describe sparse opposite-asset interventions behaviorally. Resemblance to Alpha does not establish a statistically tested equivalence or a new-asset holdout result.
- Separate original computational auditing from the public wrapper and from external replication. Explain terminal account resets, the partial final window, and the previously inspected evaluation history.

## Suggestions not used to change the frozen experiment

The automated critique proposed a larger training floor and weighted clipping quantiles. Those would be new experiments rather than editorial corrections; the pre-result floor of 20 and unweighted clipping quantiles remain unchanged. Training below 20 is explicitly blocked. Passing the floor does not guarantee a nonconstant tree fit.

The critique also questioned a narrower 28-day interval and the favorable 30-minute-delay result for the 2h transfer-current development account. Neither observation alone establishes an accounting bug. Both block analyses remain published, and direct-index bootstrap reconstruction verifies every endpoint. Public raw-market replay covers every scenario, including that delayed-execution cell; its status is recorded in `PUBLIC_VALIDATION.json`. No block length, delay, strategy or threshold was selected after this review.

## Delayed-execution diagnostic

A check of the released action records identifies four development-period checkpoint decision flips for 2h transfer-current between 15- and 30-minute execution delay:

| Checkpoint (UTC) | Asset | 15-minute prediction / action | 30-minute prediction / action |
|---|---|---|---|
| 2024-05-18 14:00 | ETHUSDT | -0.000489 / exit | +0.003921 / retain |
| 2024-07-07 16:00 | BTCUSDT | +0.000573 / retain | -0.006429 / exit |
| 2025-03-20 12:00 | ETHUSDT | +0.000474 / retain | -0.000870 / exit |
| 2025-10-13 18:00 | ETHUSDT | +0.000427 / retain | -0.003135 / exit |

The first flip retains a subsequently winning ETH episode: its candidate episode log contribution is 0.005150 under the baseline delay and 0.067816 under the longer delay. The other three flips add early interventions in episodes classified as baseline losers. Changing entry execution changes the realized entry/path state available at the later checkpoint, so this stress can change the intervention itself. These episode contributions are not additive percentage points of the two-asset portfolio. The result illustrates threshold sensitivity, not a reason to choose longer execution delay. The source records are `actions.json` and `trades.json` in the evidence directory.

## Publication tooling corrections

The first public data-replay attempt exposed 112 legacy funding manifest entries that contained paths and hashes but no URL field. The public wrapper now reconstructs their official monthly funding URLs and normalizes only the ignored working manifest. It leaves the frozen evidence and research modules unchanged. Four source archives, including the earliest BTC and latest ETH funding samples, were also fetched directly from Binance and matched their recorded SHA-256 hashes. This verifies those four retrievals, not a fresh network download of the entire cache.

Reference layout, metadata spacing and the full-contrast appendix were checked in the rendered 12-page PDF. The publication manifest hashes the final curated files; the validation record describes exactly which public checks were run.
