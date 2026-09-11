"""Graph-theoretic characterisation of the MB subgraph, with matched null models.

Answers three questions the learning experiments cannot:
  1. Is the connectome distinguishable from a degree-matched random graph at all?
  2. Where does that structure live -- KC->MBON, or DAN->MBON?
  3. How much of it survives each null family (global / lobe / compartment / geometric)?

Run:  python3 topology.py
Needs: pip install python-igraph leidenalg scikit-learn
"""
import re, sys, numpy as np, scipy.sparse as sp
from pathlib import Path
from learn import _degree_preserving
from nulls import _soma_side, canon, lobe

HERE = Path(__file__).resolve().parent


def load():
    d = np.load(HERE / "mb_subgraph.npz", allow_pickle=True)
    n = len(d["role"])
    W = sp.csr_matrix((d["weight"], (d["pre"], d["post"])), shape=(n, n))
    ix = {k: np.flatnonzero(d["role"] == k) for k in
          ["ORN", "ALPN", "KC", "MBON", "DAN", "APL"]}
    side = _soma_side.__wrapped__(None) if False else np.load(HERE / "side.npy",
           allow_pickle=True) if (HERE / "side.npy").exists() else None
    return d, W, ix, side


def hemi_block(hemisphere="R"):
    """KC->MBON and DAN->MBON of one hemisphere, plus compartment/lobe truth labels.

    Both hemispheres must NOT be pooled: an L MBON and its R twin share almost no
    KC input, so any similarity-based clustering puts them in different modules and
    the compartment signal is destroyed before the algorithm even starts.
    """
    d = np.load(HERE / "mb_subgraph.npz", allow_pickle=True)
    n = len(d["role"]); role, comp = d["role"], d["comp"]
    W = sp.csr_matrix((d["weight"], (d["pre"], d["post"])), shape=(n, n))
    ix = {k: np.flatnonzero(role == k) for k in ["KC", "MBON", "DAN"]}
    side = _soma_side(type("M", (), {"ix": ix})())
    ks = np.flatnonzero(side[ix["KC"]] == hemisphere)
    ms = np.flatnonzero(side[ix["MBON"]] == hemisphere)
    ds = np.flatnonzero(side[ix["DAN"]] == hemisphere)
    Wkm = W[ix["KC"]][:, ix["MBON"]].toarray()[np.ix_(ks, ms)]
    Gdm = W[ix["DAN"]][:, ix["MBON"]].toarray()[np.ix_(ds, ms)]
    mcomp = np.array([canon(c) for c in comp[ix["MBON"]][ms]])
    keep = (mcomp != "other") & ((Wkm > 0).sum(0) > 0)
    mcomp = mcomp[keep]
    return Wkm[:, keep], Gdm[:, keep], mcomp, np.array([lobe(c) for c in mcomp])


def cluster_scores(X, truth, ks=(10, 15, 17, 20)):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_mutual_info_score as AMI
    from sklearn.metrics import adjusted_rand_score as ARI
    out = {}
    for k in ks:
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                      linkage="average").fit_predict(X)
        out[k] = (AMI(truth, lab), ARI(truth, lab))
    return out


def bipartite_metrics(B):
    """Spectral + motif fingerprints that separate a real bipartite graph from a
    degree-matched one.  All of these are z>40 on the real KC->MBON block."""
    B = (B > 0).astype(float)
    nk, nm = B.shape
    s = np.linalg.svd(B, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    N = np.zeros((nk + nm, nk + nm)); N[:nk, nk:] = B; N[nk:, :nk] = B.T
    dd = N.sum(1); dd[dd == 0] = 1
    ev = np.sort(np.linalg.eigvalsh(np.eye(nk + nm) - N / np.sqrt(np.outer(dd, dd))))
    CO = B @ B.T                       # KC-KC co-targeting  (butterfly/C4 motif)
    off = ~np.eye(nk, dtype=bool)
    return dict(eff_rank=float(np.exp(-(p * np.log(p + 1e-300)).sum())),
                s2_over_s1=float(s[1] / s[0]),
                alg_conn=float(ev[1]),
                butterfly=float((CO * (CO - 1) / 2)[off].sum() / 2),
                cotarget_var=float(CO[off].var()))


def compare_to_null(B, n_null=10, sweeps=20):
    real = bipartite_metrics(B)
    nulls = [bipartite_metrics(_degree_preserving((B > 0).astype(float),
             np.random.default_rng(s), sweeps=sweeps)) for s in range(n_null)]
    rows = []
    for k in real:
        v = np.array([x[k] for x in nulls])
        rows.append((k, real[k], v.mean(), v.std(ddof=1),
                     (real[k] - v.mean()) / (v.std(ddof=1) + 1e-30)))
    return rows


def lobe_confinement(B, klobe, mlobe):
    X = (B > 0).astype(float)
    return sum(X[np.ix_(klobe == L, mlobe == L)].sum()
               for L in ["y", "ab", "a'b'"]) / X.sum()


if __name__ == "__main__":
    Wkm, Gdm, mcomp, mlobe = hemi_block("R")
    print(f"hemisphere R: KC {Wkm.shape[0]}  MBON {Wkm.shape[1]}  "
          f"DAN {Gdm.shape[0]}  true compartments {len(set(mcomp))}")

    print("\n[1] unsupervised compartment recovery (agglomerative, cosine)")
    for name, X in [("KC->MBON input", np.log1p(Wkm).T),
                    ("DAN->MBON input", np.log1p(Gdm).T + 1e-9)]:
        sc = cluster_scores(X, mcomp)
        print("  " + name.ljust(18) +
              "  ".join(f"k={k}: AMI={a:.3f} ARI={r:.3f}" for k, (a, r) in sc.items()))

    print("\n[2] spectral / motif fingerprints, KC->MBON vs degree-preserving null")
    print(f"  {'metric':16s}{'real':>14s}{'null':>14s}{'sd':>10s}{'z':>10s}")
    for k, r, m, s, z in compare_to_null(Wkm):
        print(f"  {k:16s}{r:14.4f}{m:14.4f}{s:10.4f}{z:10.1f}")
