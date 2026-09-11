"""Structure-preserving null families for W_km, and the conditioning test."""
import sys, re, numpy as np
from learn import LearningMB, _degree_preserving

HERE = __import__('pathlib').Path(__file__).resolve().parent
ZONE=re.compile(r"^(y|a'|B'|a|B)(\d)")
def canon(s):
    m=ZONE.match(s); return (m.group(1)+m.group(2)) if m else ('calyx' if s=='calyx' else 'other')
def lobe(c):
    if c.startswith("a'") or c.startswith("B'"): return "a'b'"
    if c.startswith('y'): return 'y'
    if c[0] in 'aB': return 'ab'
    return c

def _soma_side(m):
    """Hemisphere of every neuron, read from MaleCNS annotations (somaSide)."""
    import pyarrow.ipc as ipc
    f = HERE / 'side.npy'
    if f.exists():
        return np.load(f, allow_pickle=True)
    root = HERE.parent / 'connectome_data' / 'malecns_v1'
    ann = ipc.open_file(str(root / 'annotations.feather')).read_pandas()
    ann = ann[ann.superclass.notna()]
    lut = dict(zip(ann.bodyId.to_numpy(), ann.somaSide.fillna('').to_numpy()))
    ids = np.load(HERE / 'mb_subgraph.npz', allow_pickle=True)['ids']
    side = np.array([lut.get(int(b), '') for b in ids])
    np.save(f, side)
    return side


def blocks(m):
    side = _soma_side(m)
    kc=m.ix['KC']; mb=m.ix['MBON']
    kside=side[kc]; mside=side[mb]
    ktype=np.array([t.split('-')[0] for t in m.type[kc]])
    klobe=np.array([{'KCg':'y','KCab':'ab',"KCa'b'":"a'b'"}.get(t,'other') for t in ktype])
    mcomp=np.array([canon(c) for c in m.comp[mb]])
    mlobe=np.array([lobe(c) for c in mcomp])
    return dict(kside=kside,mside=mside,ktype=ktype,klobe=klobe,mcomp=mcomp,mlobe=mlobe)

def block_rewire(W, rowlab, collab, rng, sweeps=30):
    """Rewire only inside each (row-block, col-block) cell: block counts preserved exactly."""
    new=np.zeros_like(W)
    for rl in np.unique(rowlab):
        ri=np.flatnonzero(rowlab==rl)
        for cl in np.unique(collab):
            ci=np.flatnonzero(collab==cl)
            if len(ri)<2 or len(ci)<2: 
                new[np.ix_(ri,ci)]=W[np.ix_(ri,ci)]; continue
            new[np.ix_(ri,ci)]=_degree_preserving(W[np.ix_(ri,ci)],rng,sweeps=sweeps)
    return new

def col_block_rewire(W, collab, rng, sweeps=30):
    """Rewire within each column block only (rows free): preserves per-KC count into each block."""
    new=np.zeros_like(W)
    for cl in np.unique(collab):
        ci=np.flatnonzero(collab==cl)
        if len(ci)<2:
            new[:,ci]=W[:,ci]; continue
        new[np.ix_(np.arange(W.shape[0]),ci)]=_degree_preserving(W[:,ci],rng,sweeps=sweeps)
    return new

def make_null(m,kind,rng):
    B=blocks(m); W=m.W_km
    if kind=='real': return W.copy()
    if kind=='global': return _degree_preserving(W,rng,sweeps=30)
    if kind=='hemi':   return block_rewire(W,B['kside'],B['mside'],rng)
    if kind=='lobe':   return block_rewire(W,B['klobe'],B['mlobe'],rng)
    if kind=='hemi+lobe':
        rl=np.char.add(B['kside'].astype(str),B['klobe'].astype(str))
        cl=np.char.add(B['mside'].astype(str),B['mlobe'].astype(str))
        return block_rewire(W,rl,cl,rng)
    if kind=='comp':   return col_block_rewire(W,B['mcomp'],rng)
    if kind=='hemi+comp':
        cl=np.char.add(B['mside'].astype(str),B['mcomp'].astype(str))
        return block_rewire(W,B['kside'],cl,rng)
    if kind=='ktype+comp':
        cl=np.char.add(B['mside'].astype(str),B['mcomp'].astype(str))
        return block_rewire(W,np.char.add(B['kside'].astype(str),B['ktype'].astype(str)),cl,rng)
    raise ValueError(kind)

def run(kind,seed,us='PPL101',n_train=12,n_glom=6):
    rng=np.random.default_rng(seed)
    m=LearningMB(seed=seed,rng=rng)
    if kind!='real':
        m.W_km=make_null(m,kind,rng); m.W_km0=m.W_km.copy()
    G=m.glom_list; p=rng.permutation(len(G))
    A=[G[i] for i in p[:n_glom]]; B=[G[i] for i in p[n_glom:2*n_glom]]
    tr=m.trained_mbons(us); US=m.us(us)
    pre=m.probe(A)[tr].mean(), m.probe(B)[tr].mean()
    for _ in range(n_train):
        m.present(A,US); m.present(B,None)
    post=m.probe(A)[tr].mean(), m.probe(B)[tr].mean()
    sc=abs(pre[0])+abs(pre[1])+1e-12
    return ((post[0]-post[1])-(pre[0]-pre[1]))/sc

if __name__=='__main__':
    N=int(sys.argv[1]) if len(sys.argv)>1 else 20
    for kind in ['real','global','hemi','lobe','hemi+lobe','comp','hemi+comp','ktype+comp']:
        li=np.array([run(kind,s) for s in range(N)])
        print(f"  {kind:12s} LI = {li.mean():+.4f} ± {li.std(ddof=1):.4f}")
