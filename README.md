# flylab — associative conditioning in a connectome-derived mushroom body

A rate-based model of the *Drosophila* olfactory mushroom body, wired entirely from the
**MaleCNS v1.0** connectome (Sept 2026, HHMI Janelia FlyEM + Google Research).
It learns a differential olfactory conditioning task, and it passes the controls
that matter — odour specificity, contingency reversal, and a degree-preserving
connectome null.

**It also reports a negative result: on this task the real connectome has no
advantage over a degree-matched random rewiring. It does slightly worse.**

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

| arm | learning index | p vs paired | Cliff's δ | tier |
|---|---|---|---|---|
| **paired** | **−0.1143 ± 0.0229** | — | — | — |
| frozen weights | −0.0002 ± 0.0072 | 0.00005 | −1.000 | mechanical check |
| unpaired (US alone) | +0.0014 ± 0.0071 | 0.00005 | −1.000 | mechanical check |
| shuffled reinforcement | +0.0221 ± 0.0444 | 0.00005 | −1.000 | **real test** ✅ |
| **contingency reversal** | **+0.1356 ± 0.0357** | 0.00005 | −1.000 | **real test, sign flips** ✅ |
| degree-preserving rewire | **−0.1633 ± 0.0235** | 0.00005 | +0.900 | **null model** ⚠️ |

Permutation test, 20,000 resamples; 0.00005 is that test's floor.

**Odour specificity.** After training odour A: CS+ **−24.67% ± 2.82**, CS− −0.22% ± 0.67,
a third odour never presented −0.15% ± 0.78.

**Acquisition.** Monotone and saturating: 0 → +0.0015; 2 → −0.0541; 4 → −0.0740;
8 → −0.0949; 12 → −0.1072; 20 → −0.1202; 32 → −0.1311.

**Across compartments.** PPL101 −0.1072, PAM11 −0.1138, PAM01 −0.0784, PAM12 −0.1394.

## The negative result

| connectome | learning index |
|---|---|
| real | −0.1143 ± 0.0229 |
| rewired within KC subtype | −0.1412 ± 0.0278 |
| fully degree-preserving rewire | **−0.1633 ± 0.0235** |

**The more structure you destroy, the better it learns — monotonically.**

The null is not sloppy: KC out-degree, MBON in-degree and the weight multiset are
preserved exactly, and the observed edge overlap of 25.8% matches the configuration-model
expectation of 25.4%, so the rewiring is fully mixed.

A plausible mechanism: the real KC→MBON projection is structured, with KC subtypes
biased toward particular compartments. Random rewiring spreads each MBON's inputs evenly
across all 4,064 KCs, so any odour — which activates ~6% of them — reaches more of a
trained MBON's inputs. On a single-association task, specificity is a handicap.

This is consistent with published critiques that connectome topology advantages vanish
under matched controls. It does **not** rule out that the structure pays off on tasks this
one does not probe: parallel learning across compartments, capacity and interference,
continual learning, or anything requiring compartment-specific routing. Those are the
next experiments.

## Reproduce

```sh
python3 extract.py            # rebuild the subgraph from the release (~1 min, needs the 1.1 GB files)
python3 run_experiment.py 20  # main experiment, 6 arms × 20 seeds
python3 supplementary.py      # specificity, acquisition curve, subtype null, other compartments
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
that connectome topology beats degree-matched random topology — measured here, it does not.

The first-tier result (CS+ is depressed) follows from the rule's construction. Passing it
is not evidence. The evidence is odour specificity, the contingency reversal, and the null.

## Licences

Code: MIT (`LICENSE`). Derived connectome data: CC BY 4.0 — see `DATA_LICENSE.md`.
