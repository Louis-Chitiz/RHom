"""
RHom walkthrough 1/3 — basePCA: running and saving a single PCA
===============================================================

`basePCA` is the workhorse of RHom: it fits one PCA on a dataframe, reports the
solution, and projects new data onto it. Everything else in the package (the
grouped variant, and the whole reproducibility module) is built on top of it.

This script walks through:
    1. Loading data and separating features from metadata
    2. Fitting a PCA and reading the solution (loadings, scores, variance)
    3. The main knobs: n_components, method, corr, rotation
    4. Projecting new/held-out data with transform()
    5. Saving a tidy results folder

Run it from the repository root (the folder that contains `example_data.csv`):

    python "examples/basic example.py"
"""

import pandas as pd
from RHom import basePCA


# ---------------------------------------------------------------------------
# Configure these for your dataset
# ---------------------------------------------------------------------------
DATA_PATH = "example_data.csv"

# The columns to decompose (your measured items). List them explicitly, or build
# the list however suits your file — e.g. df.loc[:, "FirstItem":"LastItem"].columns.
FEATURE_COLS = ["item1", "item2", "item3"]   # <-- replace with your item columns

# Optional metadata columns to carry alongside the PCA scores (IDs, group labels,
# ...). Leave as [] if you have none.
METADATA_COLS = []                           # <-- e.g. ["dataset", "ID"]


# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
data = df[FEATURE_COLS + METADATA_COLS].dropna()

# basePCA only decomposes *numeric* columns. Any non-numeric metadata columns
# (string IDs, group labels) are detected automatically and carried along, so the
# PCA scores come back already joined to them — you don't have to strip them out.


# ---------------------------------------------------------------------------
# 2. Fit a PCA and inspect the solution
# ---------------------------------------------------------------------------
# verbosity=1 prints the explained variance and the top-loading items per
# component as it fits — handy for a first look.
model = basePCA(n_components=3, rotation="varimax", verbosity=1)
model.fit(data)

# The fitted solution lives on the model:
print("\nLoadings (items x components):")
print(model.loadings.round(2))

# `extra_columns` is the metadata you passed in, with one PCA_k score column per
# component appended. This is what you'd save or merge back onto your raw data.
print("\nScores joined to metadata (first rows):")
print(model.extra_columns.head())

# Eigenvalues of the retained components are also available:
print("\nEigenvalues:", model.eigenvalues.round(3))


# ---------------------------------------------------------------------------
# 3. The main knobs
# ---------------------------------------------------------------------------
# n_components : int, or "infer" to let the model choose from the eigenvalues.
# method       : "svd"   -> classic PCA (sklearn / R's prcomp). Pearson only.
#                "eigen" -> decompose a correlation matrix (SPSS / R's psych).
#                           Required if you want a non-Pearson correlation.
# corr         : "pearson" (default), "spearman" (rank, ordinal-friendly), or
#                "polychoric". Only used when method="eigen".
# rotation     : "varimax" (default), "promax", "oblimin", ... or False for none.
#
# Example: a Spearman-based solution (note method must be "eigen"):
#
#   spearman_model = basePCA(n_components=3, method="eigen",
#                            corr="spearman", rotation="varimax")
#   spearman_model.fit(data)


# ---------------------------------------------------------------------------
# 4. Project new data onto the fitted components
# ---------------------------------------------------------------------------
# transform() standardizes new rows against the *training* statistics and
# projects them onto the existing loadings — use it for held-out samples or a
# fresh dataset measured on the same items. Here we just re-project the same
# data to show the shape of the output.
projected = model.transform(data)
print("\nProjected scores (transform output, first rows):")
print(projected.head())


# ---------------------------------------------------------------------------
# 5. Save the results
# ---------------------------------------------------------------------------
# Writes loadings, scores, and a scree/variance summary into a timestamped
# subfolder of `path`, prefixed with `pathprefix` so runs are easy to tell apart.
model.save(path="results", pathprefix="basic_example_3PC")
print("\nDone — see the results/ folder.")
