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
| Stable selective router | -2.017741 | -3.603920 | -0.431562 | random 15/16, paired 0.9375, matched 14/16 | NA | NA | NA | -1.454567 | -0.301679 | 0/8 | false |
| Stable selective fallback router | -2.157969 | -3.736522 | -0.579417 | random 15/16, paired 0.8875, matched 16/16 | -2.468476 | -2.457904 | -2.881110 | -1.565944 | -0.368439 | 0/8 | false |
| Spectral-support guard | -2.186662 | -3.777646 | -0.595678 | random 13/16, paired 0.8375, matched 16/16 | NA | NA | NA | -1.446617 | -0.139976 | 3/8 | false |
| Smoothing-cap selective router | -2.193442 | -3.770138 | -0.616746 | random 16/16, paired 0.9375, matched 16/16 | pending | pending | pending | pending | pending | pending | false |

## Interpretation

The current strongest completed clean/stress result is the boundary-selective adaptive router. It improves the clean mean MSE delta from -1.289% to -2.164% and improves stress mean MSE delta from -1.391% to -2.488%.

The strongest external harm diagnostic so far is the stable selective fallback router. It keeps clean/stress gains close to the best completed result while reducing PatchTST external harm from 4/8 harmed configurations to 0/8 in the current fixture.

The current clean-table leader is the smoothing-cap selective router. It reaches -2.193% clean mean MSE delta and -0.617% clean PatchTST delta, but its stress and external evaluations are still pending.

## Current Claim Boundary

The safe claim is that HalluGuard has found a stronger internal post-processing family: boundary-aware repair plus selective residual smoothing with validation-only routing. It improves clean and stress results and substantially reduces PatchTST harm in the diagnostic fixture.

The external fixture should not yet be treated as broad generalization evidence. A larger external benchmark is needed before making that claim.
