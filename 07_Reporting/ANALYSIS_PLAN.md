# Pre-specified Analysis Plan

Study: PMST Invariant Quotient Geometry final synthetic experiment
Seed: 20260812

## Confirmatory gates

H1 Gauge exactness: median gauge-fingerprint drift <= 1e-12 and median raw-coordinate drift > 0.5.

H2 Composition completeness: exact permutation drift <= 1e-15; full (s1,s2,s3) signature distinguishes all 10 classes; total charge distinguishes fewer than 10.

H3 Quotient reconstructibility: median J drift <= 1e-12; invariant OOD accuracy >= 0.95; improvement over raw coefficients >= 0.30; median canonical reconstruction error <= 1e-12.

H4 Spectral identity: at epsilon=0.1, preserving median sector impurity <=1e-10 with nonzero (>1e-3) energy drift; breaking median impurity >0.1; sector-impurity AUC >0.95.

H5 Geometry falsification: literal recursion yields the hypercube vertex law through tested dimensions and no strict-interior central vertices.

## Statistics
- Percentile bootstrap: 1,000 resamples.
- Label permutation test: 1,000 permutations with +1 correction.
- Mann-Whitney U and Cliff's delta for H4 separation.
- Exact arithmetic identities are interpreted algebraically; Monte Carlo results are numerical confirmations, not proofs.
