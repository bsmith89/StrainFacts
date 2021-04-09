
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sfacts.genotype import genotype_linkage, prob_to_sign
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np

def calculate_clustermap_dims(nx, ny, scalex=0.15, scaley=0.02, dwidth=0.2, dheight=1.0):
    mwidth = nx * scalex
    mheight = ny * scaley
    fwidth = mwidth + dwidth
    fheight = mheight + dheight
    dendrogram_ratio = (dwidth / fwidth, dheight / fheight)
    return fwidth, fheight, dendrogram_ratio
    

def plot_genotype(gamma, linkage_kw=None, scalex=0.15, scaley=0.02, dwidth=0.2, dheight=1.0, **kwargs):
    if linkage_kw is None:
        linkage_kw = {}
    linkage, _ = genotype_linkage(gamma, **linkage_kw)
    
    gamma_t = gamma.T
    ny, nx = gamma_t.shape
    fwidth, fheight, dendrogram_ratio = calculate_clustermap_dims(
        nx, ny, scalex=scalex, scaley=scaley, dwidth=dwidth, dheight=dheight,
    )
    
    kw = dict(
        vmin=-1,
        vmax=1,
        cmap='coolwarm',
        dendrogram_ratio=dendrogram_ratio,
        col_linkage=linkage,
        figsize=(fwidth, fheight),
        xticklabels=1,
        yticklabels=0,
    )
    kw.update(kwargs)
    grid = sns.clustermap(prob_to_sign(gamma_t), **kw)
    grid.cax.set_visible(False)
    return grid
    
def plot_missing(delta, **kwargs):
    delta_t = delta.T
    ny, nx = delta_t.shape
    fwidth, fheight, dendrogram_ratio = calculate_clustermap_dims(
        nx, ny, scalex=0.15, scaley=0.02, dwidth=0.2, dheight=1.0
    )
    
    kw = dict(
        vmin=0, vmax=1, dendrogram_ratio=dendrogram_ratio, figsize=(fwidth, fheight), xticklabels=1, yticklabels=0,
    )
    kw.update(kwargs)
    grid = sns.clustermap(delta_t, **kw)
    grid.cax.set_visible(False)
    return grid
    
def plot_community(pi, scalex=0.2, scaley=0.1, dwidth=0.2, dheight=1.0, **kwargs):
    ny, nx = pi.shape
    fwidth, fheight, dendrogram_ratio = calculate_clustermap_dims(
        nx, ny, scalex=scalex, scaley=scaley, dwidth=dwidth, dheight=dheight,
    )
    
    kw = dict(
        metric='cosine', vmin=0, vmax=1, dendrogram_ratio=dendrogram_ratio, figsize=(fwidth, fheight), xticklabels=1,
    )
    kw.update(kwargs)
    grid = sns.clustermap(pi, **kw)
    grid.cax.set_visible(False)
    return grid
    
def plot_genotype_similarity(gamma, linkage_kw=None, **kwargs):
    if linkage_kw is None:
        linkage_kw = {}
    linkage, dmat = genotype_linkage(gamma, **linkage_kw)
    dmat = squareform(dmat)
    
    nx = ny = gamma.shape[0]
    fwidth, fheight, dendrogram_ratio = calculate_clustermap_dims(
        nx, ny, scalex=0.15, scaley=0.15, dwidth=0.5, dheight=0.5
    )
    
    kw = dict(
        vmin=0, vmax=1, dendrogram_ratio=dendrogram_ratio, row_linkage=linkage, col_linkage=linkage, figsize=(fwidth, fheight), xticklabels=1, yticklabels=1,
    )
    kw.update(kwargs)
    grid = sns.clustermap(1 - dmat, **kw)
    grid.cax.set_visible(False)
    return grid
    
def plot_genotype_comparison(data=None, **kwargs):
    stacked = pd.concat([
        pd.DataFrame(data[k], index=[f'{k}_{i}' for i in range(data[k].shape[0])])
        for k in data
    ])
    kw = dict(xticklabels=1)
    kw.update(kwargs)
    return plot_genotype(stacked, **kw)

def plot_community_comparison(data=None, **kwargs):
    stacked = pd.concat([
        pd.DataFrame(data[k], columns=[f'{k}_{i}' for i in range(data[k].shape[1])])
        for k in data
    ], axis=1)
    kw = dict(xticklabels=1)
    kw.update(kwargs)
    return plot_community(stacked, **kw)

def plot_loss_history(trace):
    trace = np.array(trace)
    plt.plot((trace - trace.min()))
    plt.yscale('log')
