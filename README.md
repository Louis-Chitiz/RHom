# RHom

Welcome to RHom. 

RHom is a Python-based toolbox for running, visualizing, and testing the robustness of different basic and grouped PCA.

This readme is intended for novices and assumes little-to-no prior knowledge of coding and GitHub. It will take you from installation through your first analysis, with examples. 

Before anything, make sure you have Python and a complementary code editor installed. Here's a tutorial: [Installing Python and a Code Editor](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/Installing_Python.md)

## Setting Up and Running RHom

To use this repository for your own purposes, you'll have to *fork* off a personal copy and *clone* it to your computer:

- [Fork and Clone RHom](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/Fork_RHom.md)

Once you've set up a local fork of the RHom repository on your computer, you can set up a virtual environment for using RHom:

- [Setting Up RHom](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/Set_Up_RHom.md)
    - [Updating RHom](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/updating_RHom.md)

- [Running Your First PCA Analysis](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/First_PCA_Analysis.md)

## Running a PCA: `basePCA` and `groupedPCA`

The core of RHom is the `basePCA` class. You give it a dataframe, it fits a PCA, reports the solution (loadings, scores, explained variance, wordclouds), and can project new data onto those components. Non-numeric columns (IDs, group labels) are detected automatically and carried along as metadata, so your component scores come back already joined to them.

```python
import pandas as pd
from RHom import basePCA

df = pd.read_csv("example_data.csv")
model = basePCA(n_components=4, rotation="varimax", verbosity=1)
model.fit(df)
model.save(path="results", pathprefix="my_first_pca")
```

A few options worth knowing:

- **`method`** — `"svd"` for classic PCA (as in scikit-learn / R's `prcomp`), or `"eigen"` to decompose a correlation matrix (as in SPSS / R's `psych`).
- **`corr`** — when `method="eigen"`, choose `"pearson"`, `"spearman"` (rank-based, ordinal-friendly), or `"polychoric"` (for genuinely ordinal items).
- **`rotation`** — `"varimax"` (default), `"promax"`, `"oblimin"`, and others, or `False` for none.
- **`n_components`** — an integer, or `"infer"` to let the model choose.

When your data pool several groups (studies, sites, populations) that differ in the mean or spread of each item, **`groupedPCA`** z-scores each item *within* each group before pooling, so the components reflect shared within-group structure rather than between-group differences. It is a drop-in subclass of `basePCA` with one extra argument — the column to group by:

```python
from RHom import groupedPCA
model = groupedPCA("dataset", n_components=4, rotation="varimax")
model.fit(df)
```

Worked, runnable examples for both live in [`examples/basic example.py`](examples) and [`examples/grouped example.py`](examples).

### How many components should I keep?

Before committing to a number of components, Horn's parallel analysis compares your eigenvalues against those of random data of the same shape and suggests how many to retain:

```python
from RHom.preprocessing.preliminary import parallel_analysis
result = parallel_analysis(df[feature_columns])
print(result["n_components"])
```

## The rhom Module: Testing the Robustness of Your Components

After you've generated your components, it's important to get a sense of how robustly they represent your data and how well they generalize across types of situations (e.g., different sampling environments, different participant populations, etc.).

The rhom module provides a family of analyses that assess component reliability, reproducibility, and generalizability. They share a common interface — pass your items plus, optionally, a `group` column (the variable whose levels you compare) and a `cluster` column (e.g. participant ID, kept intact during resampling so a unit never lands on both sides of a comparison). The guides below are organized by the question each function answers; a single runnable tour of all of them lives in [`examples/rhom example.py`](examples).

*How robustly do my components represent my data?*
- [Split-Half Reliability](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/split-half.md) — `splithalf` (and `splithalf_bypc` for a per-component breakdown)

*How similar are the components produced by different situations?*
- [Direct-Projection Reproducibility](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/direct-project.md) — `dir_proj` (and `dir_proj_bypc`)

*How representative are the components I get when I combine data from different situations?*
- [Omnibus-Sample Reproducibility](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/omni-sample.md) — `omni_sample` (and `omsamp_bypc`); `omni_variance` reports how much of each group's variance a pooled solution captures.

*Do my components generalize to held-out data?*
- `holdout_cv` — cross-validated reproducibility with leave-one-group-out, random K-fold, or stratified K-fold splits.

*What single set of loadings should I report?*
- `consensus_pca` — aggregates many resampled or out-of-sample PCAs into one consensus solution, with a 95% confidence interval on every loading.

Every analysis can also be run on permuted "garbage" data (`shuffle=True`) to establish a null baseline for comparison.

Answering these questions requires a metric that captures the similarity between two components (e.g., generated from different halves of the same dataset, or from separate datasets measured on the same items). The rhom module leverages two complementary metrics, plus an optional third:

- [Tucker's Congruence Coefficient (TCC): Comparing Components by Their Loadings](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/tcc.md) (reported as `phi`)
- [R-Homologue: Comparing Components by the Way They Organize Observations](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/RHom.md) (reported as `rhm`)
- Subspace similarity (`sub`), via principal angles between the two loading subspaces — rotation- and order-invariant. Add it to any analysis with `subspace=True`.

## Examples

The [`examples/`](examples) folder contains three runnable, heavily-commented walkthroughs that build on each other:

1. **`basic example.py`** — fitting, inspecting, projecting, and saving a single PCA with `basePCA`.
2. **`grouped example.py`** — pooling across groups fairly with `groupedPCA`.
3. **`rhom example.py`** — a guided tour of the whole reproducibility module.

Each script starts with a small configuration block — point `DATA_PATH` at your CSV (e.g. `example_data.csv`) and set the feature, group, and cluster columns to match — then runs top to bottom.

