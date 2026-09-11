# Corrections

## 2026-09-11 — the v1 null-model result was an artifact

The first commit of this repository reported that a degree-preserving rewiring of
KC→MBON **learns better than the real connectome**, monotonically in how much
structure is destroyed. That claim does not survive two methodological fixes.

### What was wrong

**1. The null model consumed the task random stream.**

`conditioning()` used a single `Generator` for both the model and the experiment.
Building the rewired null draws from it — 40 sweeps over 61,210 edges — so the
`rng.permutation()` that picks the odours ran from a different generator state in
the rewired arm than in the real arm.

Measured: **0 of 20 seeds gave the two arms the same odour pair.** The comparison
was not a connectome contrast at all; it was a contrast between different stimuli.

Fixed by splitting the streams: `rng` for odours and noise, `rng_null` for null
construction. Now 20/20 seeds match, and the arms are paired, so the null test is
a sign-flip permutation on per-seed differences rather than a two-sample test.

**2. The readout weighted all trained MBONs equally.**

`trained_mbons()` returns every MBON the US dopaminergic neuron touches, and v1
averaged them with equal weight — treating an MBON the DAN barely contacts the
same as its principal target. Weighting by actual innervation (`trained_weights()`)
is both the biological reading and the one under which the contrast is well posed.

### After both fixes (40 seeds, paired sign-flip permutation)

| readout | real | rewired | paired diff | p |
|---|---|---|---|---|
| equal weight (v1) | −0.1146 | −0.1426 | +0.0280 | 0.00005 |
| **DAN-synapse weighted** | **−0.4127** | **−0.4234** | **+0.0108** | **0.297** |

Across eight compartments with the corrected method (24 seeds each), **one of eight**
survives Bonferroni correction (PPL102), and the sign of the difference is mixed
(5 positive, 3 negative). **The monotone claim does not reproduce**: with the weighted
readout, real −0.4117, within-subtype rewire −0.4116, global rewire −0.4160.

### Resolution of the corrected test

Paired difference SE = 0.0101; the smallest effect detectable at 80% power is
≈ 0.0283, i.e. **6.9% of the learning magnitude**. The honest statement is
"no difference larger than about 7% of the learning magnitude is detectable",
not "there is no difference".

### What replaces it

Every null model in v1 rewired **KC→MBON** and left **ALPN→KC** untouched — so none
of them could detect structure living in the input layer, which is where the fly's
combinatorial odour code is built. Rewiring that layer instead does show an effect:
see "Where the structure actually is" in the README.

### A trap worth recording

`kc_tot`, the per-KC input normaliser, is derived from `W_pk`. Rewiring `W_pk` without
recomputing it flips the measured effect's sign — MBON-population z goes from **−12.5**
to **+2.2**, manufacturing a reverse effect out of a stale cache.
