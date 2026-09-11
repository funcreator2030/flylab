# flylab — associative conditioning in a connectome-derived mushroom body

A rate-based model of the *Drosophila* olfactory mushroom body, wired entirely from the
**MaleCNS v1.0** connectome (Sept 2026, HHMI Janelia FlyEM + Google Research).
It learns a differential olfactory conditioning task, and it passes the controls
that matter — odour specificity, contingency reversal, and a degree-preserving
connectome null.

**Where the connectome's structure shows up is not where we first looked.** In the
KC→MBON projection, no learning difference against a degree-matched rewiring is
detectable. In the **ALPN→KC** projection it is: the real wiring separates similar
odours measurably better (z = −12.5). See [`CORRECTIONS.md`](CORRECTIONS.md) — the
first version of this repository reported a stronger negative claim that turned out
to be an artifact of its own null construction.

## Why this exists

The September 2026 fly-connectome wave produced a run of demos that wire the released
connectome to games and to crypto trading. Several of them are honest about failing
their own validation gates. The most-cited one, [`nftechie/stonkfly`](https://github.com/nftechie/stonkfly),
states plainly that "Profitable learning has not been demonstrated."

Rather than patch that stack, this repository rebuilds the substrate:

| | whole-brain LIF demos | flylab |
|---|---|---|
| scope | 166,700 neurons | olfactory MB pathway, **7,824** neurons / 1,171,626 edges |
| input | price chart rendered to the retina | **odour = a subset of glomeruli**, through the real ORN→ALPN→KC pathway |
| dynamics | bistable: silent, or a self-sustaining ~527k spikes/s state | rate model with APL feedback as k-WTA |
| normalisation | none | antennal-lobe divisive gain control + per-KC input normalisation |
| stochasticity | none (no RNG anywhere) | seeded `np.random.Generator`, independent per replicate |
| compartments | 2 hand-picked (6 of 97 MBONs) | **derived from the connectome's DAN→MBON edges, zero hand labelling** |
| readout | a descending neuron pair, "right = buy" by fiat | evoked MBON response in the trained compartment, double-differenced |
| plastic synapses | 7,835 | **61,210** |

## The pathway

ORN 2,635 (**53 glomeruli**) → ALPN 686 → KC 4,064 → MBON 97; DAN 340; APL 2.
Every edge comes from the release. Each KC receives a **median of 6 ALPN inputs**,
matching the canonical value for the fly.

## Gate 1 — odour separability

The whole-brain LIF demos fail here: two completely disjoint odours drive nearly the
same Kenyon cells (reported Jaccard 0.685–0.890), so there is nothing left to learn.

| check | result | criterion |
|---|---|---|
| Jaccard(KC_A, KC_B), disjoint glomeruli | **0.013** | < 0.3 ✅ |
| leave-one-out decoding, 20 odours × 8 reps, 5% noise | **100.0%** | > 95% ✅ |
| KC sparsity | 6.00% | 3–10% ✅ (calibrated, not emergent) |
| non-degeneracy | one odour drives 73–87% of KCs; k-WTA keeps 244 | selection is doing real work |

Combinatorial structure is graded and monotone — sharing 0→6 glomeruli gives
Jaccard 0.013 / 0.051 / 0.113 / 0.195 / 0.368 / 0.551 / 1.000.

## Gate 2 — the compartment map falls out of the connectome

No hand labels. A DAN and an MBON share a compartment if the DAN synapses onto that MBON.
This recovers the published map:

| DAN | top MBON from the connectome | literature |
|---|---|---|
| PPL101 | MBON11 (γ1pedc) w=1391 | γ1pedc ✅ |
| PAM11 | MBON07 (α1) w=515 | α1 ✅ |
| PAM01 | MBON01 (γ5β'2a) w=816 | γ5 ✅ |
| PAM12 | MBON09 (γ3β'1) w=592 | γ3 ✅ |
| PAM07 | MBON05 (γ4) w=439 | γ4 ✅ |
| PPL103 | MBON32 (γ2) w=530 | γ2α'1 ✅ |

## Learning rule

Canonical mushroom-body form — dopamine depresses a KC→MBON synapse when it coincides
with presynaptic KC activity, compartment by compartment. The update is heterosynaptic:
no term reads MBON activity.

```
dw[i,m] = -eta * r_kc[i] * dan_drive[m] * w[i,m]
w      += rho * (w0 - w)
```

## Results — differential conditioning, 20 independent seeds per arm

Readout is the evoked MBON response in the trained compartment, weighted by how
strongly the US dopaminergic neuron actually innervates each MBON.

| arm | learning index | tier |
|---|---|---|
| **paired** | **−0.3907 ± 0.0551** | — |
| frozen weights | ≈ 0 | mechanical check |
| unpaired (US alone) | ≈ 0 | mechanical check |
| shuffled reinforcement | abolished | **real test** ✅ |
| **contingency reversal** | **sign flips** | **real test** ✅ |

**Odour specificity.** After training odour A: CS+ **−24.67% ± 2.82**, CS− −0.22% ± 0.67,
a third odour never presented −0.15% ± 0.78.

**Acquisition.** Monotone and saturating: 0 → +0.0009; 2 → −0.2255; 4 → −0.3013;
8 → −0.3651; 12 → −0.3907; 20 → −0.4122; 32 → −0.4208.

**Across compartments.** PPL101 −0.3907, PAM11 −0.3775, PAM01 −0.3406, PAM12 −0.3206.

## The KC→MBON null: no detectable difference

Real connectome vs degree-preserving rewire, 40 seeds, arms matched on stimuli,
paired sign-flip permutation:

| readout | real | rewired | paired diff | p |
|---|---|---|---|---|
| equal weight | −0.1146 | −0.1426 | +0.0280 | 0.00005 |
| **DAN-synapse weighted** | **−0.4127** | **−0.4234** | **+0.0108** | **0.297** |

Across eight compartments, one of eight survives Bonferroni (PPL102) and the signs
are mixed (5 positive, 3 negative). Paired SE is 0.0101, so the smallest effect
detectable at 80% power is ≈ 6.9% of the learning magnitude: **no difference larger
than about 7% is detectable**, which is not the same as no difference.

## Where the structure actually is

Every null above rewires KC→MBON. The fly's combinatorial odour code is built one
layer earlier, in ALPN→KC — so rewire *that* instead, preserving degrees, and measure
how well similar odours (sharing 3 of 6 glomeruli) stay apart. 40 odour pairs,
10 null instances:

| | KC Jaccard | KC corr | **MBON population corr** | sparsity |
|---|---|---|---|---|
| real | 0.2181 | 0.3972 | **0.8954** | 0.0600 |
| ALPN→KC rewired | 0.2327 | 0.4040 | **0.9553** | 0.0600 |
| **z (real vs null)** | **−4.4** | −0.7 | **−12.5** | identical |

Lower correlation is better separation. Sparsity is identical to four decimals, so
this is not a sparsity confound. **The real connectome keeps similar odours further
apart than a degree-matched rewiring of the same projection.**

The learning index is blind to this: rewiring ALPN→KC leaves the LI statistically
unchanged. The structure is real, large, and measurable in the representation — and a
single scalar behavioural readout cannot see it.

**Caution.** `kc_tot`, the per-KC input normaliser, is derived from `W_pk`. Rewiring
`W_pk` without recomputing it flips the sign of this result (z = +2.2), inventing a
reverse effect from a stale cache. `separability.py` runs both so the trap is visible.

## Reproduce

```sh
python3 extract.py            # rebuild the subgraph from the release (~1 min, needs the 1.1 GB files)
python3 run_experiment.py 20  # main experiment, 6 arms × 20 seeds
python3 supplementary.py      # specificity, acquisition curve, subtype null, other compartments
python3 null_test.py          # KC->MBON null, paired, both readouts, 8 compartments
python3 separability.py       # ALPN->KC null, pattern separation of similar odours
```

`mb_subgraph.npz` ships with the repository, so the experiments run without the download.
To rebuild it, fetch the three MaleCNS v1.0 files into `connectome_data/malecns_v1/`:

```
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

Requires `numpy`, `scipy`, `pyarrow`, `pandas`.

## What this supports, and what it does not

**Supported.** This connectome plus this rule sustains compartment-specific,
odour-specific, contingency-sensitive differential conditioning.

**Calibrated by us, not discovered.** The k-WTA target of 6%, the form of the antennal-lobe
normalisation, `eta` and `rho`, and the choice of a rate model.

**Not supported.** That the fly learns this way; that any of this transfers to trading;
that connectome topology beats degree-matched random topology *on the learning index* —
no such difference is detectable above ~7% of the learning magnitude. The separation
result is a representational claim, not a behavioural one.

The first-tier result (CS+ is depressed) follows from the rule's construction. Passing it
is not evidence. The evidence is odour specificity, the contingency reversal, and the null.

## Licences

Code: MIT (`LICENSE`). Derived connectome data: CC BY 4.0 — see `DATA_LICENSE.md`.
