"""Extract the olfactory mushroom-body subgraph from MaleCNS v1.0.

ORN -> ALPN -> KC -> MBON, plus DAN (modulatory) and APL (global inhibition).
Nothing here is invented: every edge comes from the released connectome.
"""
import re, json, hashlib, numpy as np, pyarrow.ipc as ipc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "connectome_data" / "malecns_v1"
OUT  = Path(__file__).resolve().parent / "mb_subgraph.npz"


def compartment(tok: str) -> str:
    """MBON/DAN *instance* names carry the compartment in parentheses.

    'MBON01(y5B\'2a)_R' -> 'y5B\'2a';  'MBON05(y4>y1y2)_R' -> 'y4'
    ('>' and '<' are projection operators: the left side is the compartment
    the dendrite actually sits in.)  Letters follow the dataset's ASCII
    convention: y=gamma, B=beta, a=alpha, ' = prime.
    """
    m = re.search(r"\(([^)]*)\)", tok or "")
    if not m:
        return "unknown"
    return re.split(r"[><]", m.group(1))[0].strip() or "unknown"


def main():
    ann = ipc.open_file(str(DATA / "annotations.feather")).read_pandas()
    ann = ann[ann.superclass.notna()].reset_index(drop=True)
    ids = ann.bodyId.to_numpy()
    pos = {int(b): i for i, b in enumerate(ids)}

    typ = ann.type.fillna("").to_numpy()
    cls = ann["class"].fillna("").to_numpy()
    inst = ann.instance.fillna("").to_numpy()

    groups = {
        "ORN":  np.flatnonzero(np.char.startswith(typ.astype(str), "ORN")),
        "ALPN": np.flatnonzero(cls == "ALPN"),
        "KC":   np.flatnonzero(np.char.startswith(typ.astype(str), "KC")),
        "MBON": np.flatnonzero(cls == "MBON"),
        "DAN":  np.flatnonzero(cls == "DAN"),
        "APL":  np.flatnonzero(np.char.startswith(typ.astype(str), "APL")),
    }
    for k, v in groups.items():
        print(f"  {k:5s} {len(v):>6,}")

    keep = np.unique(np.concatenate(list(groups.values())))
    local = {int(g): i for i, g in enumerate(keep)}

    ed = ipc.open_file(str(DATA / "edges.feather")).read_pandas()
    ed = ed[ed.body_pre.isin(pos) & ed.body_post.isin(pos)]
    gpre = ed.body_pre.map(pos).to_numpy()
    gpost = ed.body_post.map(pos).to_numpy()
    m = np.isin(gpre, keep) & np.isin(gpost, keep)
    pre = np.array([local[int(x)] for x in gpre[m]], dtype=np.int32)
    post = np.array([local[int(x)] for x in gpost[m]], dtype=np.int32)
    w = ed.weight.to_numpy()[m].astype(np.float32)
    print(f"  subgraph: {len(keep):,} neurons, {len(pre):,} edges")

    role = np.array(["other"] * len(keep), dtype=object)
    for k, v in groups.items():
        role[[local[int(g)] for g in v]] = k

    sub_type = typ[keep].astype(str)
    sub_inst = inst[keep].astype(str)
    comp = np.array([compartment(i) if r in ("MBON", "DAN") else ""
                     for i, r in zip(sub_inst, role)], dtype=object)

    # Glomerulus labels: ORN types read "ORN_DA1"; ALPN types read "DA1_lPN".
    def glom_of(t, r):
        if r == "ORN":
            return t.split("_", 1)[1] if "_" in t else ""
        if r == "ALPN":
            return t.split("_", 1)[0] if "_" in t and not t.startswith("CB") else ""
        return ""
    glom = np.array([glom_of(t, r) for t, r in zip(sub_type, role)], dtype=object)

    np.savez_compressed(
        OUT, ids=ids[keep], pre=pre, post=post, weight=w,
        role=role.astype(str), type=sub_type, instance=sub_inst,
        comp=comp.astype(str), glom=glom.astype(str),
    )
    h = hashlib.sha256(OUT.read_bytes()).hexdigest()
    meta = {
        "source": "MaleCNS v1.0 (CC-BY 4.0), https://male-cns.janelia.org/download/",
        "paper": "10.1016/j.cell.2026.08.015",
        "neurons": int(len(keep)), "edges": int(len(pre)),
        "counts": {k: int(len(v)) for k, v in groups.items()},
        "sha256": h,
    }
    (OUT.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
