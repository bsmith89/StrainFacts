import pandas as pd


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
    from skbio.diversity import beta_diversity

    if discretized:
        genotype = world.genotype.discretized()
    else:
        genotype = world.genotype

    dm = genotype.pdist()
    dm = DistanceMatrix(dm, dm.index.astype(str))
    tree = neighbor_joining(dm).root_at_midpoint()
    out = beta_diversity(
        "weighted_unifrac",
        counts=(world.community.data.to_pandas() * coef),
        ids=world.sample.astype(str).values,
        otu_ids=world.strain.astype(str).values,
        normalized=True,
        tree=tree,
    ).to_data_frame()
    out.index = world.sample
    out.columns = world.sample
    return out
