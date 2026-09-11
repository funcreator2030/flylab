"""Does the connectome's ALPN->KC projection separate similar odours better
than a degree-matched rewiring of that same projection?

Every null model in the first version rewired KC->MBON and left the input layer
untouched -- so it could not have detected structure that lives in ALPN->KC.
"""
import numpy as np
from learn import LearningMB, _degree_preserving
from model import jaccard


def pattern_separation(m, n_pairs=40, shared=3, n_glom=6, seed=0):
    """Similar-odour pairs: `shared` glomeruli in common out of n_glom."""
    rng = np.random.default_rng(seed)
    G = m.glom_list
    jac, kcorr, mcorr = [], [], []
    for _ in range(n_pairs):
        p = rng.permutation(len(G))
        A = [G[i] for i in p[:n_glom]]
        B = [G[i] for i in p[:shared]] + [G[i] for i in p[n_glom:n_glom + (n_glom - shared)]]
        fa, fb = m.forward(m.orn_drive(A)), m.forward(m.orn_drive(B))
        jac.append(jaccard(m.kc_active(fa["kc"]), m.kc_active(fb["kc"])))
        kcorr.append(np.corrcoef(fa["kc"], fb["kc"])[0, 1])
        mcorr.append(np.corrcoef(fa["mbon"], fb["mbon"])[0, 1])
    return dict(jaccard=np.mean(jac), kc_corr=np.mean(kcorr), mbon_corr=np.mean(mcorr),
                sparsity=float((fa["kc"] > 0).mean()))


def rewire_input(m, rng, recompute_norm=True):
    m.W_pk = _degree_preserving(m.W_pk, rng, sweeps=40)
    if recompute_norm:                       # the trap: kc_tot is derived from W_pk
        m.kc_tot = np.maximum(m.W_pk.sum(0), 1e-9)
    return m


if __name__ == "__main__":
    real = pattern_separation(LearningMB(seed=0))
    print("=== 相似气味（共享 3/6 嗅小球）的模式分离，40 对 ===")
    print(f"{'':22s} {'KC Jaccard':>11s} {'KC相关':>9s} {'MBON群体相关':>13s} {'稀疏度':>8s}")
    print(f"{'真实连接组':22s} {real['jaccard']:11.4f} {real['kc_corr']:9.4f} "
          f"{real['mbon_corr']:13.4f} {real['sparsity']:8.4f}")

    for tag, recompute in [("W_pk重连(正确:重算kc_tot)", True),
                           ("W_pk重连(错误:沿用旧kc_tot)", False)]:
        rows = []
        for i in range(10):
            m = LearningMB(seed=0)
            rewire_input(m, np.random.default_rng(5000 + i), recompute_norm=recompute)
            rows.append(pattern_separation(m))
        for key, lab in [("jaccard", None), ("mbon_corr", None)]:
            pass
        mj = np.array([r["jaccard"] for r in rows]); mm = np.array([r["mbon_corr"] for r in rows])
        mk = np.array([r["kc_corr"] for r in rows]); ms = np.array([r["sparsity"] for r in rows])
        zj = (real["jaccard"] - mj.mean()) / mj.std(ddof=1)
        zm = (real["mbon_corr"] - mm.mean()) / mm.std(ddof=1)
        zk = (real["kc_corr"] - mk.mean()) / mk.std(ddof=1)
        print(f"{tag:22s} {mj.mean():11.4f} {mk.mean():9.4f} {mm.mean():13.4f} {ms.mean():8.4f}")
        print(f"{'  → z (真实 vs 零模型)':22s} {zj:11.1f} {zk:9.1f} {zm:13.1f}")
