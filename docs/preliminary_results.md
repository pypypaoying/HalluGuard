# Preliminary Results

This document summarizes the current research snapshot for HalluGuard. The numbers come from local prototype experiments and should be interpreted as preliminary rather than final benchmark claims.

The reported metric is MSE delta percentage relative to the uncorrected forecast. Negative values indicate lower MSE after correction.

## Evaluation Setup

The current evaluation uses DLinear and PatchTST forecasting outputs across multiple horizons. Three types of checks are used:

- **Clean benchmark**: standard forecast correction table.
- **Stress benchmark**: synthetic perturbation suite covering boundary discontinuities, trend drift, slope breaks, delayed level shifts, high-frequency perturbations, and variance shifts.
- **External fixture**: compact diagnostic set used to check whether a correction method causes harm outside the main clean/stress table.

Controls include random-action routing, matched sparse smoothing, dominant-action checks, and test-threshold leakage checks.

## Main Result Snapshot

| Method | Clean MSE delta | Clean DLinear delta | Clean PatchTST delta | Clean evidence | Stress MSE delta | Boundary stress | High-frequency stress | External MSE delta | External PatchTST delta | PatchTST harmed | Leakage |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| Adaptive router baseline | -1.289319 | -2.281087 | -0.297552 | random 15/16, paired 0.95, matched 12/16 | -1.390795 | -1.482901 | -1.282335 | -0.930070 | +0.004431 | 4/8 | false |
| Capped logistic router | -1.657706 | -2.832699 | -0.482713 | random 15/16, paired 0.90, matched 15/16 | -1.920196 | -1.817744 | -2.422435 | -1.154303 | -0.135139 | 3/8 | false |
| Boundary-selective adaptive router | -2.163965 | -3.774653 | -0.553277 | random 15/16, paired 0.9125, matched 15/16 | -2.487840 | -2.501779 | -2.890815 | -1.310251 | -0.045764 | 2/8 | false |
| Stable selective fallback router | -2.157969 | -3.736522 | -0.579417 | random 15/16, paired 0.8875, matched 16/16 | -2.468476 | -2.457904 | -2.881110 | -1.565944 | -0.368439 | 0/8 | false |
| Spectral-support guard | -2.186662 | -3.777646 | -0.595678 | random 13/16, paired 0.8375, matched 16/16 | NA | NA | NA | -1.446617 | -0.139976 | 3/8 | false |
| Smoothing-cap selective router | -2.193442 | -3.770138 | -0.616746 | random 16/16, paired 0.9375, matched 16/16 | -2.508625 | -2.485757 | -2.895440 | -1.317923 | -0.065154 | 2/8 | false |
| Stable smoothing-cap guard | -2.135378 | -3.699258 | -0.571499 | random 15/16, paired 0.8875, matched 16/16 | -2.463475 | -2.447172 | -2.888124 | -1.567057 | -0.366358 | 0/8 | false |
| Conditional stable-cap guard | -2.181291 | -3.753660 | -0.608921 | random 16/16, paired 0.9500, matched 16/16 | -2.504963 | -2.476395 | -2.894392 | -1.469623 | -0.171489 | 2/8 | false |

## Interpretation

The current strongest clean/stress result is the smoothing-cap selective router. It improves the clean mean MSE delta from -1.289% to -2.193% and the stress mean MSE delta from -1.391% to -2.509%.

The strongest external harm diagnostic so far is the stable smoothing-cap guard. It reduces PatchTST external harm from 4/8 harmed configurations to 0/8 in the current fixture, but gives up some clean and stress performance compared with the smoothing-cap selective router.

The conditional stable-cap guard is a compromise candidate. It preserves most of the clean/stress gains while improving external behavior compared with the main clean/stress leader, but it does not fully remove PatchTST harm.

## Current Claim Boundary

The safe claim is that HalluGuard has found a stronger internal post-processing family: boundary-aware repair plus selective residual smoothing with validation-only routing and confidence-capped smoothing deployment. It improves clean and stress results, and stable-forecast guards can substantially reduce PatchTST harm in the diagnostic fixture.

The external fixture should not yet be treated as broad generalization evidence. A larger external benchmark is needed before making that claim.
