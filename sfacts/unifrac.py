import pandas as pd
from scipy.spatial.distance import pdist


def neighbor_joining(dm):
    from skbio.tree import nj as _neighbor_joining

    tree = _neighbor_joining(dm)
    if pd.Series(dm.ids).astype(str).str.contains("_").any():
        unrenamed_ids = pd.Series(
            dm.ids, index=[name.replace("_", " ") for name in dm.ids]
        )
        for node in tree.tips():
            node.name = unrenamed_ids[node.name]
    return tree


def unifrac_pdist(world, coef=1e6, discretized=False):
    from skbio import DistanceMatrix
    from skbio.diversity.beta import weighted_unifrac

    if discretized:
        genotypes = world.genotypes.discretized()
    else:
        genotypes = world.genotypes

    dm = genotypes.pdist()
    dm = DistanceMatrix(dm, dm.index.astype(str))
    tree = neighbor_joining(dm).root_at_midpoint()
    return pdist(
        world.communities.values * coef,
        metric=weighted_unifrac,
        otu_ids=world.strain.values.astype(str),
        tree=tree,
        validate=False,
    )
