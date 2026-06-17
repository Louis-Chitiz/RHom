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

The rhom module provides a family of analyses that assess component reliability, reproducibility, and generalizability. They share a common interface — pass your items plus, optionally, a `group` column (the variable whose levels you compare), a `cluster` column (e.g. participant ID, kept intact during resampling so a unit never lands on both sides of a comparison), and a `groupby` column (to run the analysis on a `groupedPCA` solution — see [Grouped solutions](#grouped-solutions-the-groupby-argument) below). The guides below are organized by the question each function answers; a single runnable tour of all of them lives in [`examples/rhom example.py`](examples).

*How robustly do my components represent my data?*
- [Split-Half Reliability](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/split-half.md) — `splithalf` (and `splithalf_bypc` for a per-component breakdown)

*How similar are the components produced by different situations?*
- [Direct-Projection Reproducibility](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/direct-project.md) — `dir_proj` (and `dir_proj_bypc`)

*How representative are the components I get when I combine data from different situations?*
- [Omnibus-Sample Reproducibility](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/omni-sample.md) — `omni_sample` (and `omsamp_bypc`); `omni_variance` reports how much of each group's variance a pooled solution captures.

*Do my components generalize to held-out data?*
- `holdout_cv` (and `holdout_bypc`) — cross-validated reproducibility with leave-one-group-out, random K-fold, or stratified K-fold splits.

*What single set of loadings should I report?*
- `consensus_pca` — aggregates many resampled or out-of-sample PCAs into one consensus solution, with a 95% confidence interval on every loading.

Every similarity-metric analysis also estimates a **chance reference by default** (`null=True`): it permutes the data (destroying cross-variable structure), re-runs the comparison, and overlays the chance level + 95% CI on the figure — a dashed reference line and shaded band on the bar/component plots, or a chance-centred colour scale plus annotation on the `dir_proj` heatmaps. This matters because the chance level for these metrics is *not* zero (they pick a best match, so even random components score positively), so "is my reproducibility above chance?" can only be judged against it. Set `null=False` to skip, or `null_reps=` (default 200) to change the number of permutations.

Answering these questions requires a metric that captures the similarity between two components (e.g., generated from different halves of the same dataset, or from separate datasets measured on the same items). The rhom module leverages two complementary metrics, plus an optional third:

- [Tucker's Congruence Coefficient (TCC): Comparing Components by Their Loadings](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/tcc.md) (reported as `phi`)
- [R-Homologue: Comparing Components by the Way They Organize Observations](https://github.com/Louis-Chitiz/Rhom/blob/main/tutorials/RHom.md) (reported as `rhm`)
- Subspace similarity (`sub`), via principal angles between the two loading subspaces — rotation- and order-invariant. Add it to any analysis with `subspace=True`.

### Grouped solutions: the `groupby` argument

Every rhom analysis accepts an optional `groupby` column that switches the underlying decomposition from `basePCA` to `groupedPCA` semantics — i.e. each variable is z-scored *within each level of `groupby`* before decomposing, so the components reflect shared within-group structure rather than between-group differences (see `groupedPCA` above for the motivation):

```python
# Split-half reliability of a grouped solution, standardizing within study:
splithalf(df=df, npc=4, cluster="ID", groupby="dataset",
          method="eigen", corr="spearman")

# Held-out CV of a grouped solution:
holdout_cv(df=df, folds=5, npc=4, groupby="site")
```

Two things worth knowing:

- **`groupby` is independent of `group`.** `group` is the variable whose levels you *compare* / iterate over / partition by; `groupby` is the nuisance variable you *standardize within*. The guiding rule is to set `groupby` to match the model you actually report — if your solution is a `groupedPCA` within some variable, group-standardize by that same variable so the reproducibility estimate describes the model you're using. The one nuance is `omni_sample`: with `groupby=group` it measures the reproducibility of your grouped solution (correct if that's your model), but it no longer tests whether *raw* between-group differences justify pooling in the first place — for that "should I blend these at all?" question, run it with `groupby=None` (or compare the two; the gap is informative). For `dir_proj`, `groupby=group` has no real effect (each comparison is already within one group), so use a separate nuisance grouping there.
- **Standardization is leakage-free.** The within-group z-scoring is applied to each resample (fold, half, bootstrap sample) *on its own rows, after the split* — never to the whole dataset before splitting — so reproducibility estimates are not inflated by group-level information shared across a split. This mirrors how each side of a comparison is fit as an independent grouped solution.

## Examples

The [`examples/`](examples) folder contains three runnable, heavily-commented walkthroughs that build on each other:

1. **`basic example.py`** — fitting, inspecting, projecting, and saving a single PCA with `basePCA`.
2. **`grouped example.py`** — pooling across groups fairly with `groupedPCA`.
3. **`rhom example.py`** — a guided tour of the whole reproducibility module.

Each script starts with a small configuration block — point `DATA_PATH` at your CSV (e.g. `example_data.csv`) and set the feature, group, and cluster columns to match — then runs top to bottom.

