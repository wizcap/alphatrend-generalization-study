# Generalization Limits of Adaptive AlphaTrend Exit Models

**Research preprint v1.0.0 - not peer reviewed.**

This repository publishes the final fixed generalization comparison from the NEATTRADE research program. It does **not** establish a deployable profitable trading strategy. The source, trained research models, derived results and reproduction tools accompany the paper.

- [Read the paper](paper/manuscript.md) / [PDF](paper/alphatrend-generalization.pdf)
- [繁體中文結論](paper/conclusion_zh-TW.md)
- [Frozen method](evidence/study015/study015_method.md)
- [All results](evidence/study015/report.json) / [asset costs and funding](evidence/study015/asset_results.csv)
- [Original computational verification](evidence/study015/verification.json)
- [Original automated-review disposition](evidence/study015/study015_review_resolution.md) / [manuscript review notes](paper/review_notes.md)
- [Public release validation](PUBLIC_VALIDATION.json)

## Conclusion

The proposed compact, robust, temporal and cross-asset exit models did not produce sufficiently consistent evidence of improved generalization over the existing monthly learner. Some return estimates improve in one period while worsening in another. All history, including 2026, had already been inspected in earlier research. No deployment candidate was selected.

| Policy | 2024-2025 net return | Known 2026 through Aug 31 |
|---|---:|---:|
| 2h original monthly | 13.07% | 5.80% |
| 2h Ridge8 | 28.73% | 4.43% |
| 4h original monthly | 38.13% | -3.02% |
| 4h temporal mean | 13.89% | -2.09% |

Returns include seven basis points of friction per traded side, 15-minute execution delay and actual historical mark-price funding. The 2026 window ends at 2026-08-31 20:00 UTC and is not annualized. The full report includes every policy and both execution stresses; the rows above illustrate period-dependent tradeoffs rather than select winners.

The frozen study evaluated ten policies at 2h/4h, fit 704 new monthly heads, retained 64 frozen reference versions and replayed 720 asset accounts. Its original audit passed 303 tests and checked 23,010 actions, 12,210 episode attributions, 3,600 diagnostic rows and 40 adjusted bootstrap intervals. None of those intervals is wholly positive. These counts describe computational checks, not independent statistical samples or external peer review.

## Reproduce

Use Python 3.13 and an isolated environment. The recorded runtime is Python 3.13.14; exact package versions are in the requirements chain and original protocol. Model pickles are publisher-generated, version-sensitive research artifacts; the wrapper verifies the public manifest before loading them.

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -r requirements-public.txt
python -m pytest -q
python tools/reproduce.py
python tools/reproduce.py --intervals
python tools/reproduce.py --models
python tools/reproduce.py --download
python tools/reproduce.py --market
```

- Default: reconstruct every published asset/portfolio return and drawdown, annual and per-asset summaries, and the 20 contrast point estimates from released derived curves.
- `--intervals`: additionally regenerate all 40 development-only circular-block bootstrap intervals using direct sample indices.
- `--models`: independently reconstruct cohorts, preprocessing and all 704 new model fits, including manual tree prediction and weighted Ridge checks.
- `--download`: retrieve the 355 official archives using recorded SHA-256 hashes and provider URLs. Legacy funding URLs are reconstructed from their recorded filenames using the original monthly archive convention. No Binance API key is required. Provider archives are excluded from Git.
- `--market`: use those local archives to reconstruct the continuous labels, decision state and 720 complete asset curves with the independent policy/accounting implementation. It requires the prior download and has no live-order functionality.

Checks write to ignored `reproduction/`. They do not overwrite frozen evidence. The original research modules are retained byte-for-byte for attribution and inspection; some legacy module entry points refer to unpublished operational bookkeeping. Use `tools/reproduce.py` for this curated release. It deliberately does not claim to recheck unpublished shadow-controller state or unrelated earlier experiments.

## Scope and interpretation

Only BTCUSDT and ETHUSDT USD-M perpetual futures are used. Each policy may intervene once 24 hours after an AlphaTrend entry. The model cannot add entries; it is latched at entry and consumes only naturally resolved training labels known before the monthly boundary minus seven days. Transfer fitting removes the target asset before preprocessing. No order-book inputs are used.

Each half-year account is terminally closed and reinitialized under the inherited protocol. Monthly updates do not flatten positions. Reported drawdown comes from the complete stitched curve and is not a live-account drawdown. The fixed eight-feature/model restriction is a package comparison; it does not identify the effect of a single removed indicator. Both temporal ensemble members and monthly training versions overlap heavily.

This publication preserves a negative result and its limitations. A public GitHub preprint is not journal publication, acceptance, or a DOI registration. Automated Claude critiques and separate computational implementations are not external scientific replication.

## Contents and integrity

`PUBLIC_MANIFEST.json` hashes the curated release. Original study hashes and verification records remain in `evidence/study015/`; their references to excluded operational records are provenance, not promises that those private files are published. `neattrade/` and `tests/` contain the frozen research code. `references/alphatrend/` preserves the original attributed MPL-2.0 Pine source.

To rebuild the paper, install `requirements-paper.txt` in a separate document environment and run `python tools/build_paper.py`. Source numerical results are never refit by the document builder.

See [NOTICE.md](NOTICE.md) for attribution and licensing and [CITATION.cff](CITATION.cff) for citation metadata. Code uses MPL-2.0; the original manuscript and generated figures use CC BY 4.0. Underlying provider data and cited works retain their own terms.
