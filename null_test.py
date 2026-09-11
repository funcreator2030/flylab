"""Real connectome vs degree-preserving null, with arms matched on stimuli.

Because both arms now draw the same odours from the same task stream, the
comparison is paired: the right test is a sign-flip permutation on the
per-seed difference, not a two-sample test.
"""
import numpy as np
from learn import conditioning


def paired_permutation(diff, n=20000, rng=None):
    rng = rng or np.random.default_rng(0)
    obs = abs(diff.mean())
    cnt = 0
    for _ in range(n):
        s = rng.choice([-1.0, 1.0], size=len(diff))
        if abs((diff * s).mean()) >= obs - 1e-15:
            cnt += 1
    return (cnt + 1) / (n + 1)


def compare(us="PPL101", n=40, readout="dan_weighted", n_train=12):
    real = np.array([conditioning(seed=s, us_dan=us, readout=readout, n_train=n_train)["LI"]
                     for s in range(n)])
    null = np.array([conditioning(seed=s, us_dan=us, readout=readout, n_train=n_train,
                                  rewire=True)["LI"] for s in range(n)])
    d = real - null
    p = paired_permutation(d, rng=np.random.default_rng(1))
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(real=real.mean(), null=null.mean(), diff=d.mean(), se=se, p=p,
                mde80=2.8 * se)


if __name__ == "__main__":
    print("=== 修复后：真实连接组 vs 保度重连（配对，40 种子）===")
    print(f"{'读出':16s} {'真实':>9s} {'重连':>9s} {'配对差':>9s} {'p':>9s}")
    for ro in ["equal", "dan_weighted"]:
        r = compare(readout=ro)
        lab = "等权(旧)" if ro == "equal" else "DAN突触加权(新)"
        print(f"{lab:16s} {r['real']:+9.4f} {r['null']:+9.4f} {r['diff']:+9.4f} {r['p']:9.5f}")

    print("\n=== 跨 8 个隔室复现（DAN 加权，24 种子）===")
    print(f"{'US':9s} {'真实':>9s} {'重连':>9s} {'配对差':>9s} {'p':>8s}")
    res = {}
    for us in ["PPL101", "PPL102", "PAM01", "PAM02", "PAM05", "PAM08", "PAM11", "PAM12"]:
        r = compare(us=us, n=24)
        res[us] = r
        star = " *" if r["p"] < 0.05 / 8 else ""
        print(f"{us:9s} {r['real']:+9.4f} {r['null']:+9.4f} {r['diff']:+9.4f} {r['p']:8.4f}{star}")
    sig = sum(1 for r in res.values() if r["p"] < 0.05 / 8)
    pos = sum(1 for r in res.values() if r["diff"] > 0)
    print(f"\n  Bonferroni(×8) 后显著: {sig}/8   配对差为正: {pos}/8")

    r = compare(n=40)
    print(f"\n=== 装置分辨率 ===")
    print(f"  配对差 SE = {r['se']:.4f}；80% 功效可测效应 ≈ {r['mde80']:.4f}"
          f" = 学习幅度的 {100*r['mde80']/abs(r['real']):.1f}%")
