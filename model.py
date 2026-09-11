"""Connectome-derived mushroom-body model (rate-based, MaleCNS v1.0).

Pathway is entirely from the released connectome:
    ORN -> ALPN -> KC -> MBON,  with APL global feedback inhibition.

Three mechanisms are *model choices*, declared here rather than buried:
  1. antennal-lobe divisive normalisation (Olsen & Wilson 2010 form),
  2. per-KC input normalisation -- without it the same "intrinsically loud"
     KCs win for every odour, which is the separability failure we are fixing,
  3. APL feedback as k-winner-take-all, solved by bisection on a global
     threshold to hit a target sparsity.
"""
import numpy as np, scipy.sparse as sp
from pathlib import Path

HERE = Path(__file__).resolve().parent


class MB:
    def __init__(self, npz=None, seed=0, target_sparsity=0.06,
                 kc_norm=True, al_norm=True, pn_norm=True, rng=None):
        d = np.load(npz or (HERE / "mb_subgraph.npz"))
        self.role, self.type, self.comp, self.glom = d["role"], d["type"], d["comp"], d["glom"]
        n = len(self.role)
        self.n = n
        pre, post, w = d["pre"], d["post"], d["weight"]
        self.W = sp.csr_matrix((w, (pre, post)), shape=(n, n))
        self.ix = {k: np.flatnonzero(self.role == k)
                   for k in ["ORN", "ALPN", "KC", "MBON", "DAN", "APL"]}
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.k = target_sparsity
        self.kc_norm, self.al_norm, self.pn_norm = kc_norm, al_norm, pn_norm

        self.W_op = self._block("ORN", "ALPN")
        self.W_pk = self._block("ALPN", "KC")
        self.W_km = self._block("KC", "MBON")
        self.W_km0 = self.W_km.copy()                      # baseline efficacy
        # normalisers (guard against zero-input cells)
        self.pn_tot = np.maximum(np.asarray(self.W_op.sum(0)).ravel(), 1e-9)
        self.kc_tot = np.maximum(np.asarray(self.W_pk.sum(0)).ravel(), 1e-9)
        self.glom_list = sorted(set(self.glom[self.ix["ORN"]]) - {""})

    def _block(self, a, b):
        return self.W[self.ix[a]][:, self.ix[b]].toarray().astype(np.float64)

    # ---------------- forward ----------------
    def orn_drive(self, glomeruli, intensity=1.0, noise=0.0):
        g = self.glom[self.ix["ORN"]]
        r = np.isin(g, list(glomeruli)).astype(float) * intensity
        if noise:
            r = np.maximum(r + self.rng.normal(0, noise, r.shape), 0.0)
        return r

    def forward(self, r_orn, noise=0.0):
        raw_pn = r_orn @ self.W_op
        if self.pn_norm:
            raw_pn = raw_pn / self.pn_tot          # identity, not synapse count
        if self.al_norm:                            # AL divisive gain control
            raw_pn = raw_pn / (0.05 + raw_pn.mean())
        if noise:
            raw_pn = np.maximum(raw_pn + self.rng.normal(0, noise * raw_pn.mean(), raw_pn.shape), 0)

        raw_kc = raw_pn @ self.W_pk
        if self.kc_norm:
            raw_kc = raw_kc / self.kc_tot
        if noise:
            raw_kc = raw_kc + self.rng.normal(0, noise * raw_kc.std(), raw_kc.shape)

        r_kc = self._apl(raw_kc)
        r_mbon = r_kc @ self.W_km
        return dict(pn=raw_pn, kc=r_kc, mbon=r_mbon, kc_raw=raw_kc)

    def _apl(self, raw):
        """APL feedback inhibition, solved as k-WTA on a single global threshold."""
        live = raw[np.isfinite(raw)]
        if live.max() <= 0:
            return np.zeros_like(raw)
        theta = np.quantile(raw, 1.0 - self.k)
        return np.maximum(raw - theta, 0.0)

    # ---------------- helpers ----------------
    def kc_active(self, kc):
        return np.flatnonzero(kc > 0)

    def sparsity(self, kc):
        return float((kc > 0).mean())


def jaccard(a, b):
    A, B = set(a.tolist()), set(b.tolist())
    return len(A & B) / max(len(A | B), 1)
