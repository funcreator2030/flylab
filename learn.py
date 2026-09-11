"""Dopamine-gated plasticity and differential conditioning on the MB model.

Learning rule (canonical mushroom-body form, Hige 2015 / Cohn 2015):
a KC->MBON synapse is *depressed* when presynaptic KC activity coincides with
dopamine in that synapse's compartment.  Compartment membership is not hand
labelled -- it is read off the connectome's DAN->MBON edges, which recover the
published compartment map exactly (PPL101->MBON11/gamma1pedc, PAM11->MBON07/alpha1, ...).

    dw[i,m] = -eta * r_kc[i] * dan_drive[m] * w[i,m]      (depression)
    w      += rho * (w0 - w)                              (slow recovery)

No term in the rule reads MBON activity: the update is heterosynaptic, as the
biology requires.
"""
import numpy as np, scipy.sparse as sp
from model import MB


class LearningMB(MB):
    def __init__(self, *a, eta=0.35, rho=0.002, **kw):
        super().__init__(*a, **kw)
        n = self.n
        W = self.W
        self.G = W[self.ix["DAN"]][:, self.ix["MBON"]].toarray()      # 340 x 97
        gs = self.G.sum(0); gs[gs == 0] = 1.0
        self.Gn = self.G / gs                                        # column-normalised
        self.eta, self.rho = eta, rho
        self.dan_type = self.type[self.ix["DAN"]]
        self.mbon_type = self.type[self.ix["MBON"]]
        self.mbon_comp = self.comp[self.ix["MBON"]]

    # ---- US: activate a named DAN population ----
    def us(self, dan_type, level=1.0):
        a = np.zeros(len(self.ix["DAN"]))
        a[self.dan_type == dan_type] = level
        return a

    def dan_drive(self, dan_act):
        return dan_act @ self.Gn                                     # per-MBON dopamine

    def trained_mbons(self, dan_type, top=None):
        """MBONs in the compartment(s) the US DAN innervates -- from the connectome."""
        tot = self.G[self.dan_type == dan_type].sum(0)
        ix = np.flatnonzero(tot > 0)
        if top:
            ix = ix[np.argsort(-tot[ix])[:top]]
        return ix

    # ---- one presentation ----
    def present(self, odor, dan_act=None, learn=True, noise=0.03):
        f = self.forward(self.orn_drive(odor, noise=noise), noise=noise)
        if learn and dan_act is not None and dan_act.any():
            drive = self.dan_drive(dan_act)
            dw = -self.eta * np.outer(f["kc"], drive) * self.W_km
            self.W_km += dw
            np.clip(self.W_km, 0.0, None, out=self.W_km)
        if learn:
            self.W_km += self.rho * (self.W_km0 - self.W_km)
        return f

    def probe(self, odor, reps=4, noise=0.03):
        """Read MBON response without any plasticity."""
        out = []
        for _ in range(reps):
            f = self.forward(self.orn_drive(odor, noise=noise), noise=noise)
            out.append(f["mbon"])
        return np.mean(out, axis=0)


def conditioning(seed=0, us_dan="PPL101", n_train=12, arm="paired",
                 rewire=False, eta=0.35, sparsity=0.06, n_glom=6):
    """Differential conditioning: CS+ paired with US, CS- unpaired.

    arm: paired | frozen | shuffled | swapped | unpaired
    """
    rng = np.random.default_rng(seed)
    m = LearningMB(seed=seed, rng=rng, target_sparsity=sparsity, eta=eta)
    if rewire:
        m.W_km = _degree_preserving(m.W_km, rng, sweeps=40)
        m.W_km0 = m.W_km.copy()

    G = m.glom_list
    p = rng.permutation(len(G))
    A = [G[i] for i in p[:n_glom]]                 # odour A  (label is fixed)
    B = [G[i] for i in p[n_glom:2 * n_glom]]       # odour B  (disjoint)
    # "swapped" reverses the CONTINGENCY only: B becomes the reinforced odour,
    # while the endpoint stays LI = (A - B).  A real reversal must flip LI's sign.
    cs_plus, cs_minus = (B, A) if arm == "swapped" else (A, B)

    tr = m.trained_mbons(us_dan)
    pre = m.probe(A)[tr].mean(), m.probe(B)[tr].mean()

    US = m.us(us_dan)
    for t in range(n_train):
        if arm == "frozen":
            m.present(cs_plus, US, learn=False); m.present(cs_minus, None, learn=False)
        elif arm == "shuffled":
            # same total dopamine, decoupled from odour identity
            m.present(cs_plus, US if rng.random() < 0.5 else None)
            m.present(cs_minus, US if rng.random() < 0.5 else None)
        elif arm == "unpaired":
            m.present(cs_plus, None); m.present(cs_minus, None)
            m.present([], US)                      # US alone, no odour
        else:                                      # paired / swapped
            m.present(cs_plus, US); m.present(cs_minus, None)

    post = m.probe(A)[tr].mean(), m.probe(B)[tr].mean()
    d_pre, d_post = pre[0] - pre[1], post[0] - post[1]
    scale = abs(pre[0]) + abs(pre[1]) + 1e-12
    return dict(arm=arm, seed=seed, LI=(d_post - d_pre) / scale,
                pre_A=pre[0], pre_B=pre[1], post_A=post[0], post_B=post[1],
                n_trained_mbon=len(tr))


def _degree_preserving(W, rng, sweeps=40):
    """Bipartite double-edge swap: preserves BOTH KC out-degree and MBON in-degree.

    Pick edge pairs (k1,m1),(k2,m2) and rewire to (k1,m2),(k2,m1) when neither
    target edge already exists.  Weights ride along with their edge, so the
    weight multiset is preserved too.
    """
    r, c = np.nonzero(W > 0)
    r, c = r.astype(np.int64), c.astype(np.int64)
    vals = W[r, c].copy()
    E = len(r)
    ncol = W.shape[1]
    present = set((int(a) * ncol + int(b)) for a, b in zip(r, c))
    for _ in range(sweeps):
        i = rng.permutation(E); j = rng.permutation(E)
        for a, b in zip(i, j):
            if a == b:
                continue
            k1, m1, k2, m2 = r[a], c[a], r[b], c[b]
            if k1 == k2 or m1 == m2:
                continue
            n1, n2 = k1 * ncol + m2, k2 * ncol + m1
            if n1 in present or n2 in present:
                continue
            present.discard(k1 * ncol + m1); present.discard(k2 * ncol + m2)
            present.add(n1); present.add(n2)
            c[a], c[b] = m2, m1
    new = np.zeros_like(W)
    new[r, c] = vals
    return new
