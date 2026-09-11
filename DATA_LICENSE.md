# Data attribution

`mb_subgraph.npz` is a derived subgraph of the **MaleCNS v1.0** connectome,
redistributed here under the terms of its licence.

- Dataset: MaleCNS v1.0, HHMI Janelia FlyEM / Cambridge Connectomics Group / Google Research
- Licence: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- Source: https://male-cns.janelia.org/download/
- Paper: https://doi.org/10.1016/j.cell.2026.08.015

The derived file contains only the olfactory mushroom-body pathway
(ORN, ALPN, KC, MBON, DAN, APL: 7,824 neurons, 1,171,626 edges) together with
type, instance, compartment and glomerulus labels from the release's
annotation table. It is a *derived interpretation*, not an official dataset
product. Cite the dataset and its paper when publishing results built on it.

Code in this repository is MIT licensed (see `LICENSE`). The data licence
above governs `mb_subgraph.npz` and anything derived from it.
