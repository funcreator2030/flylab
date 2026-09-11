"""Differential-conditioning experiment with the full control battery."""
import json, sys, numpy as np
from pathlib import Path
from learn import conditioning

ARMS = [
    ("paired",   dict(arm="paired"),                 "配对 (CS+ 与 US 配对)"),
    ("frozen",   dict(arm="frozen"),                 "冻结权重 (无可塑性)"),
    ("shuffled", dict(arm="shuffled"),               "打乱强化 (多巴胺与气味解耦)"),
    ("unpaired", dict(arm="unpaired"),               "非配对 (US 单独给)"),
    ("swapped",  dict(arm="swapped"),                "列联反转 (改训 B)"),
    ("rewired",  dict(arm="paired", rewire=True),    "保度重连连接组"),
]


def permutation_p(a, b, n=20000, rng=None):
    rng = rng or np.random.default_rng(0)
    obs = abs(np.mean(a) - np.mean(b))
    pool = np.concatenate([a, b]); k = len(a)
    cnt = 0
    for _ in range(n):
        p = rng.permutation(pool)
        if abs(p[:k].mean() - p[k:].mean()) >= obs - 1e-15:
            cnt += 1
    return (cnt + 1) / (n + 1)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def main(n_seeds=20, us="PPL101", n_train=12, out="results.json"):
    res = {}
    for key, kw, label in ARMS:
        rows = [conditioning(seed=s, us_dan=us, n_train=n_train, **kw) for s in range(n_seeds)]
        li = np.array([r["LI"] for r in rows])
        res[key] = dict(label=label, LI=li.tolist(), mean=float(li.mean()),
                        sd=float(li.std(ddof=1)), rows=rows)
        print(f"  {label:28s} LI = {li.mean():+.4f} ± {li.std(ddof=1):.4f}")

    base = np.array(res["paired"]["LI"])
    print("\n=== 对照检验（vs 配对臂，置换检验 20000 次）===")
    stats = {}
    for key, kw, label in ARMS[1:]:
        arr = np.array(res[key]["LI"])
        p = permutation_p(base, arr, rng=np.random.default_rng(42))
        d = cliffs_delta(base, arr)
        stats[key] = dict(p=p, cliffs_delta=float(d))
        verdict = "配对臂显著不同 ✓" if p < 0.01 else "无法区分 ✗"
        print(f"  paired vs {key:10s} p = {p:.5f}  Cliff's δ = {d:+.3f}   {verdict}")
    res["_stats"] = stats
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n_seeds=n)
