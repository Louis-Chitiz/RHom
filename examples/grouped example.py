"""
RHom walkthrough 2/3 — groupedPCA: pooling across groups, fairly
================================================================

When your data come from several groups (different studies, sites, populations,
task conditions...), the groups often differ in the mean and spread of each
item. A plain PCA on the pooled data can then turn those between-group
differences into components — you end up describing "which group a row came
from" rather than the within-group structure you actually care about.

`groupedPCA` removes that artifact. It z-scores every item *within each group*
first, then pools the rows back together for a single decomposition. The
resulting components reflect covariance that is shared within groups, not
differences between them.

It is a drop-in subclass of `basePCA` — same fit / transform / save interface —
with one extra required argument: the column to group by.

Run from the repository root:

    python "examples/grouped example.py"
"""

import pandas as pd
from RHom import groupedPCA


# ---------------------------------------------------------------------------
# Configure these for your dataset
# ---------------------------------------------------------------------------
DATA_PATH = "example_data.csv"

# The columns to decompose (your measured items).
FEATURE_COLS = ["item1", "item2", "item3"]   # <-- replace with your item columns

# The column to group by — the nuisance grouping we want to standardize away
# before decomposing (e.g. study, site, or population).
GROUP_COL = "group"                          # <-- replace with your grouping column

# Optional extra metadata columns to carry along (IDs, etc.). Leave as [] if none.
METADATA_COLS = []                           # <-- e.g. ["ID"]


# ---------------------------------------------------------------------------
# 1. Load data with a grouping column
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
data = df[FEATURE_COLS + [GROUP_COL] + METADATA_COLS].dropna()


# ---------------------------------------------------------------------------
# 2. Fit a grouped PCA
# ---------------------------------------------------------------------------
# The first argument is the grouping column. Each group's items are z-scored
# against that group's own mean and SD before everything is pooled and
# decomposed once. (Any metadata columns ride along, as in the basic example.)
model = groupedPCA(GROUP_COL, n_components=3, rotation="varimax", verbosity=1)
model.fit(data)

print("\nGrouped loadings (items x components):")
print(model.loadings.round(2))

# Scores come back joined to the metadata, just like basePCA.
print("\nScores joined to metadata (first rows):")
print(model.extra_columns.head())


# ---------------------------------------------------------------------------
# 3. Projecting new data
# ---------------------------------------------------------------------------
# transform() reuses the per-group scalers fitted above, so new rows from a
# known group are standardized with that group's training statistics before
# projection. A group never seen during fit() is z-scored on its own (with a
# warning).
projected = model.transform(data)
print("\nProjected scores (first rows):")
print(projected.head())


# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
# Same as basePCA. (The output folder is prefixed "grouped_" automatically.)
model.save(path="results", pathprefix="grouped_example_3PC")
print("\nDone — see the results/ folder.")


# ---------------------------------------------------------------------------
# basePCA vs groupedPCA — which one?
# ---------------------------------------------------------------------------
# Use basePCA when your rows are exchangeable and you want the structure of the
# data as-is. Use groupedPCA when rows are nested in groups whose location/scale
# differences you consider nuisance, and you want components driven by shared
# within-group structure. If the two give very different loadings, that gap is
# itself informative — it means between-group differences were shaping the plain
# PCA. The reproducibility tools in walkthrough 3 quantify exactly that.
#
# Note: groupedPCA standardizes within groups itself, so it always decomposes a
# correlation-like (z-scored) matrix. It exposes n_components, rotation, and
# method, but not `corr` — for Spearman/polychoric grouped solutions you'd
# pre-rank the data before fitting.
