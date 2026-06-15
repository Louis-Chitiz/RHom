"""
RHom walkthrough 3/3 — the rhom module: how robust are your components?
=======================================================================

A PCA always returns *a* solution. The rhom module asks the harder question:
would you get the *same* solution again — from another half of your sample,
another study, or after pooling several sources together? Each builtin answers
one version of that question, all using the same few similarity metrics:

    phi  (Tucker's Congruence Coefficient)  -- do the loadings agree?
    rhm  (R-Homologue)                      -- do the component *scores* agree,
                                               i.e. do the components order the
                                               observations the same way?
    sub  (subspace similarity, optional)    -- do the components span the same
                                               subspace (rotation/order-invariant)?

This script is a guided tour. It runs top to bottom and writes a results folder
per analysis (CSV + PNG). With the settings below it takes a few minutes; see
the "PERFORMANCE" notes to speed it up or scale it up for a final run.

Map of the tour:
    0. Setup
    1. How many components?            parallel_analysis
    2. Reliability (one sample)        splithalf / splithalf_bypc
    3. Reproducibility across groups   dir_proj / dir_proj_bypc
    4. Blending sources together       omni_sample / omsamp_bypc / omni_variance
    5. Out-of-sample generalization    holdout_cv / holdout_bypc
    6. One consensus solution          consensus_pca
    7. The null baseline               shuffle=True
    8. Re-plotting saved results

Run from the repository root:

    python "examples/rhom example.py"
"""

import matplotlib
# Non-interactive backend: figures are written to disk but no window pops up, so
# the script runs unattended. Delete this line (or switch to e.g. "TkAgg") if
# you'd rather see each figure interactively as it's produced.
matplotlib.use("Agg")

import pandas as pd

from RHom import (
    splithalf, splithalf_bypc,
    dir_proj, dir_proj_bypc,
    omni_sample, omsamp_bypc, omni_variance,
    holdout_cv, holdout_bypc,
    consensus_pca,
)


# ---------------------------------------------------------------------------
# 0. Setup — configure these for your dataset
# ---------------------------------------------------------------------------
# The dataframe you pass to a builtin should contain only the items to decompose
# plus, optionally, two bookkeeping columns:
#
#   group   : the variable whose levels you compare (e.g. study / site / population)
#   cluster : the unit to keep intact when resampling (e.g. participant ID).
#             rhom never splits a cluster across both sides of a comparison,
#             which prevents leakage with repeated-measures / nested data.
#
DATA_PATH = "example_data.csv"
FEATURE_COLS = ["item1", "item2", "item3"]   # <-- replace with your item columns
GROUP_COL = "group"                          # <-- the variable whose levels you compare
CLUSTER_COL = "ID"                           # <-- the unit kept intact when resampling

df = pd.read_csv(DATA_PATH)
data = df[FEATURE_COLS + [GROUP_COL, CLUSTER_COL]].dropna()

# House settings for this tour. If your items are bounded / ordinal ratings, the
# eigen decomposition of a Spearman correlation matrix is a good default; for
# continuous data the defaults (method="svd", corr="pearson" — Pearson PCA) are
# usually what you want.
NPC = 4              # components to retain (see section 1)
METHOD = "eigen"
CORR = "spearman"
ROTATION = "varimax"

# PERFORMANCE: bootstrap/fold counts drive runtime. These are deliberately low so
# the whole script finishes quickly. For reported results use BOOT >= 1000.
BOOT = 200
FOLDS = 5


# ---------------------------------------------------------------------------
# 1. How many components? — parallel_analysis
# ---------------------------------------------------------------------------
# Horn's parallel analysis compares your eigenvalues against those of random data
# of the same shape and suggests how many components to keep. Do this before any
# reproducibility analysis, since every builtin needs an `npc`.
from RHom.preprocessing.preliminary import parallel_analysis

pa = parallel_analysis(data[FEATURE_COLS], seed=0, title="Parallel analysis")
print(f"Parallel analysis suggests {pa['n_components']} components "
      f"(this tour uses NPC={NPC}).")
pa["figure"].savefig("results/parallel_analysis.png", bbox_inches="tight", dpi=150)


# ---------------------------------------------------------------------------
# 2. Reliability within one sample — splithalf
# ---------------------------------------------------------------------------
# "If I split my data in half, do both halves give me the same components?"
# Bootstrap repeatedly draws two random halves and scores their similarity.
#
#   - group=None  -> treat the whole dataset as one sample.
#   - stratify    -> draw each half proportionally from every source, so one
#                    study can't dominate a half. (Use group= instead to run the
#                    analysis separately within each level.)
#   - subspace=True adds the rotation-invariant subspace metric.
splithalf(
    df=data, npc=NPC, method=METHOD, corr=CORR, rotation=ROTATION,
    cluster=CLUSTER_COL, stratify=GROUP_COL, boot=BOOT, subspace=True,
    file_prefix="tour_splithalf",
)

# Per-component breakdown: which individual components are reliable? Useful when
# you expect early components to be solid and later ones shaky.
splithalf_bypc(
    df=data, npc=NPC, method=METHOD, corr=CORR, rotation=ROTATION,
    cluster=CLUSTER_COL, stratify=GROUP_COL, boot=BOOT,
    file_prefix="tour_splithalf",
)


# ---------------------------------------------------------------------------
# 3. Reproducibility across groups — dir_proj
# ---------------------------------------------------------------------------
# "Do different studies recover the same components?" Direct projection takes
# every pair of groups, fits a PCA in each, and measures how well one group's
# components reproduce in the other. Output includes pairwise heatmaps.
dir_proj(
    df=data, group=GROUP_COL, npc=NPC, method=METHOD, corr=CORR, rotation=ROTATION,
    cluster=CLUSTER_COL, folds=FOLDS, subspace=True,
    file_prefix="tour_dirproj",
)

# By-component version: a grid of per-component pairwise heatmaps, so you can see, say,
# that PC1 reproduces everywhere but PC4 only between two studies.
dir_proj_bypc(
    df=data, group=GROUP_COL, npc=NPC, method=METHOD, corr=CORR, rotation=ROTATION,
    cluster=CLUSTER_COL, folds=FOLDS,
    file_prefix="tour_dirproj",
)


# ---------------------------------------------------------------------------
# 4. Blending sources together — omni_sample / omni_variance
# ---------------------------------------------------------------------------
# "If I pool everything into one 'omnibus' solution, how faithfully does it
# represent each individual source?" This is the key question before combining
# datasets. Each source is compared against the pooled components.
omni_sample(
    df=data, group=GROUP_COL, npc=NPC, method=METHOD, corr=CORR, rotation=ROTATION,
    cluster=CLUSTER_COL, boot=BOOT, subspace=True,
    file_prefix="tour_omni",
)

# Per-component view of the same comparison.
omsamp_bypc(
    df=data, group=GROUP_COL, npc=NPC, method=METHOD, corr=CORR, rotation=ROTATION,
    cluster=CLUSTER_COL, folds=FOLDS,
    file_prefix="tour_omni",
)

# A descriptive complement: fit ONE pooled PCA, then report how much of each
# group's variance those pooled components actually capture (stacked-bar plot
# with a chance baseline). Deterministic — no bootstrap, so it's fast.
omni_variance(
    df=data, group=GROUP_COL, npc=NPC, method=METHOD, corr=CORR, rotation=ROTATION,
    cluster=CLUSTER_COL,
    file_prefix="tour_omni",
)


# ---------------------------------------------------------------------------
# 5. Out-of-sample generalization — holdout_cv
# ---------------------------------------------------------------------------
# "Do components fit on held-out data match components fit on the training data?"
# Three modes, chosen by which arguments you pass:
#
#   group= only          -> leave-one-group-out (hold out each source in turn)
#   folds= only          -> random K-fold (cluster-aware via cluster=)
#   group= AND folds=    -> stratified K-fold (every fold contains every source)
#
#   boot= (+ optional folds=/groups= AND folds=)   -> bootstrap K-fold/stratified K-fold
#
# Leave-one-source-out — the strongest "does this generalize to a new study?" test:
holdout_cv(
    df=data, group=GROUP_COL, npc=NPC, method=METHOD, corr=CORR, rotation=ROTATION,
    cluster=CLUSTER_COL, subspace=True,
    file_prefix="tour_holdout_logo",
)

# Stratified 5-fold — keeps all sources in every fold:
holdout_cv(
    df=data, group=GROUP_COL, folds=FOLDS, npc=NPC,
    method=METHOD, corr=CORR, rotation=ROTATION, cluster=CLUSTER_COL,
    file_prefix="tour_holdout_strat",
)

# Per-component view of the stratified 5-fold.
holdout_bypc(
    df=data, group=GROUP_COL, folds=FOLDS, npc=NPC,
    method=METHOD, corr=CORR, rotation=ROTATION, cluster=CLUSTER_COL,
    file_prefix="tour_holdout_strat"
)


# ---------------------------------------------------------------------------
# 6. One consensus solution — consensus_pca
# ---------------------------------------------------------------------------
# Once you trust the components are reproducible, consensus_pca gives you a single
# set of loadings to report: it fits many resampled/out-of-sample PCAs, aligns
# each to a shared anchor, and averages them — with a 95% CI on every loading.
#
#   boot= (+ optional cluster=/stratify=)   -> bootstrap consensus
#   group=                                  -> leave-one-group-out consensus
#   folds= / group=+folds=                  -> K-fold / stratified K-fold
#
# Cluster-aware, source-stratified bootstrap consensus:
consensus_pca(
    df=data, boot=BOOT, npc=NPC, method=METHOD, corr=CORR, rotation=ROTATION,
    cluster=CLUSTER_COL, stratify=GROUP_COL,
    file_prefix="tour_consensus_boot",
)


# ---------------------------------------------------------------------------
# 7. The null baseline — shuffle=True
# ---------------------------------------------------------------------------
# How high would these scores be by chance? Pass shuffle=True to permute each
# item independently (destroying cross-item structure) and re-run. Compare the
# real numbers above against this noise floor to judge what counts as "good".
splithalf(
    df=data, npc=NPC, method=METHOD, corr=CORR, rotation=ROTATION,
    cluster=CLUSTER_COL, stratify=GROUP_COL, boot=BOOT, shuffle=True,
    file_prefix="tour_splithalf_null",
)


# ---------------------------------------------------------------------------
# 8. Re-plotting saved results
# ---------------------------------------------------------------------------
# Every builtin returns its results dataframe AND writes a CSV, so you can
# regenerate or restyle a figure later without re-running the analysis. The
# plotting helpers live in RHom.visualization.rhomplots.
#
#   import pandas as pd
#   from RHom.visualization.rhomplots import plot_omni
#   ho = pd.read_csv("results/tour_holdout_logo/"
#                    "tour_holdout_logo_holdout_cv_<N>D_4PC.csv")
#   fig = plot_omni(ho, group="fold", metric="rhm")
#   fig.savefig("results/tour_holdout_logo/replot_rhm.png", bbox_inches="tight")
#
# Filenames encode the feature count and component count — e.g. with 10 items and
# NPC=4 the file ends in _10D_4PC. Check the results folder for the exact name.

print("\nTour complete — every analysis wrote a folder under results/.")
