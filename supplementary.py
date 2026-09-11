"""Supplementary experiments: specificity, learning curve, subtype null, second US."""
import numpy as np
from learn import LearningMB, _degree_preserving, conditioning


def specificity(seed=0, us="PPL101", n_train=12, n_glom=6):
    """After training A, is a novel odour C left untouched?"""
    rng = np.random.default_rng(seed)
    m = LearningMB(seed=seed, rng=rng)
    G = m.glom_list; p = rng.permutation(len(G))
    A = [G[i] for i in p[:n_glom]]
    B = [G[i] for i in p[n_glom:2*n_glom]]
    C = [G[i] for i in p[2*n_glom:3*n_glom]]          # never presented
    tr = m.trained_mbons(us); US = m.us(us)
    pre = {k: m.probe(o)[tr].mean() for k, o in [("A", A), ("B", B), ("C", C)]}
    for _ in range(n_train):
        m.present(A, US); m.present(B, None)
    post = {k: m.probe(o)[tr].mean() for k, o in [("A", A), ("B", B), ("C", C)]}
    return {k: (post[k] - pre[k]) / pre[k] for k in pre}


def subtype_rewire(W, kc_type, rng, sweeps=40):
    """Rewire only within KC subtype: preserves subtype->MBON composition."""
    new = W.copy()
    for st in np.unique(kc_type):
        ix = np.flatnonzero(kc_type == st)
        if len(ix) < 4:
            continue
        new[np.ix_(ix, np.arange(W.shape[1]))] = _degree_preserving(
            W[np.ix_(ix, np.arange(W.shape[1]))], rng, sweeps=sweeps)
    return new


def conditioning_subtype(seed=0, us="PPL101", n_train=12, n_glom=6):
    rng = np.random.default_rng(seed)
    m = LearningMB(seed=seed, rng=rng)
    kt = np.array([t.split("-")[0] for t in m.type[m.ix["KC"]]])
    m.W_km = subtype_rewire(m.W_km, kt, rng); m.W_km0 = m.W_km.copy()
    G = m.glom_list; p = rng.permutation(len(G))
    A = [G[i] for i in p[:n_glom]]; B = [G[i] for i in p[n_glom:2*n_glom]]
    tr = m.trained_mbons(us); US = m.us(us)
    pre = m.probe(A)[tr].mean(), m.probe(B)[tr].mean()
    for _ in range(n_train):
        m.present(A, US); m.present(B, None)
    post = m.probe(A)[tr].mean(), m.probe(B)[tr].mean()
    return ((post[0]-post[1]) - (pre[0]-pre[1])) / (abs(pre[0])+abs(pre[1])+1e-12)


if __name__ == "__main__":
    N = 20
    print("=== 补充1: 气味特异性（训练 A，B 与从未出现的 C 应不受影响）===")
    rows = [specificity(s) for s in range(N)]
    for k in ["A", "B", "C"]:
        v = np.array([r[k] for r in rows])
        lab = {"A": "CS+ (训练)", "B": "CS- (出现过未配对)", "C": "C (从未出现)"}[k]
        print(f"  {lab:22s} MBON 响应变化 {100*v.mean():+7.2f}% ± {100*v.std(ddof=1):.2f}")

    print("\n=== 补充2: 学习曲线 ===")
    for nt in [0, 2, 4, 8, 12, 20, 32]:
        li = np.array([conditioning(seed=s, n_train=nt)["LI"] for s in range(10)])
        print(f"  训练 {nt:>2d} 试次 → LI {li.mean():+.4f} ± {li.std(ddof=1):.4f}")

    print("\n=== 补充3: 亚型内重连（保住 KC亚型→MBON 组成）vs 全局重连 ===")
    st = np.array([conditioning_subtype(seed=s) for s in range(N)])
    real = np.array([conditioning(seed=s)["LI"] for s in range(N)])
    full = np.array([conditioning(seed=s, rewire=True)["LI"] for s in range(N)])
    print(f"  真实连接组      LI {real.mean():+.4f} ± {real.std(ddof=1):.4f}")
    print(f"  亚型内重连      LI {st.mean():+.4f} ± {st.std(ddof=1):.4f}")
    print(f"  全局保度重连    LI {full.mean():+.4f} ± {full.std(ddof=1):.4f}")

    print("\n=== 补充4: 换一个隔室的 US（PAM11 → α1，奖赏型）===")
    for us in ["PPL101", "PAM11", "PAM01", "PAM12"]:
        li = np.array([conditioning(seed=s, us_dan=us)["LI"] for s in range(10)])
        m0 = LearningMB(seed=0)
        print(f"  US={us:7s} 受训MBON {len(m0.trained_mbons(us)):>3d} 个  LI {li.mean():+.4f} ± {li.std(ddof=1):.4f}")
