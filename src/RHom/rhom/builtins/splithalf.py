"""
Split-half reliability analyses.

``splithalf`` is the aggregate version (bootstrap mean reproducibility); ``splithalf_bypc``
is the per-component breakdown anchored against a fixed per-sample reference frame
(same architecture as ``dir_proj_bypc``, with the splithalf resampling protocol).
"""
from ..._deps import pd, np, randint, plt

import os
import copy

from ...core.base_pca import basePCA
from ...preprocessing.preliminary import _check_rank
from ...preprocessing.data_utils import group_standardize
from ...io.save import setupanalysis

from ..metrics import RHom
from ..resampling import pair_cv
from ..bootstrap import BootstrapEngine

from ._reporting import _build_row, _display_stats, _export_report, _chance_reference


def splithalf(df=None, group=None, npc=None, method='svd', rotation='varimax', corr='pearson',
              boot=1000, save=True, display=False, null=True, null_reps=200, cluster=None, stratify=None,
              groupby=None, subspace=False, progress=True, path='results', file_prefix=randint(10000, 99999)):
    """
    Split-Half Reliability
    ----------------------
    This function conducts a bootstrapped split-half reliability analysis
    on your dataframe. It can do so on a full dataset, or at each level of a
    grouping variable. It simply bootstrap reassigns random halves of the data
    into two subsets and computes their component similarity based on:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)

    Parameters
    ----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        cluster: str, default=None
            Optional level-2 / clustering column (e.g. participant ID). When provided,
            whole clusters are kept together in every resample, fold, and split, so a unit
            never appears on both sides of a comparison. Prevents leakage and
            pseudoreplication with nested data. Requires at least `folds` distinct clusters
            per group where cross-validation is used.

        groupby: str, default=None
            Optional nuisance-grouping column for groupedPCA-style decomposition: each
            variable is z-scored within each level of this column before decomposing
            (per resample, on its own rows -- leakage-free), so components reflect
            within-group covariance rather than between-group differences. Independent
            of ``group`` / ``cluster`` / ``stratify``.

        stratify: str, default=None
            Optional stratification column for whole-dataset splithalf. When provided
            (and ``group`` is None), each bootstrap half is drawn proportionally from
            every level of this column so a small source can't be over-represented in
            one half. Mutually exclusive with ``group`` (which means "iterate per level"
            rather than "balance across levels"). Composes with ``cluster``: within each
            stratum, whole clusters are kept on one side.

        subspace: bool, default=False
            If True, also report subspace similarity via principal angles between the two
            loading subspaces (sub_* columns; rotation- and order-invariant, in [0, 1]).

        corr: str, default="pearson"
            Which correlation matrix to decompose under `method='eigen'`: "pearson",
            "spearman" (rank correlation, ordinal-friendly), or "polychoric" (latent
            correlation behind ordinal items via Olsson 1979 MLE; meaningful only for
            genuinely ordinal data and noticeably slower). Ignored under `method='svd'`,
            which is Pearson-only; pass `method='eigen'` to switch correlation type.

        npc: int, default=None
            Number of components to extract per solution.

        rotation: str, default="varimax"
            Rotation method to be performed on referent. "none" for no rotation.

        boot: int, default=1000
            Number of bootstrap samples to generate 95% confidence intervals.

        save: bool, default=True
            Save outputted split-half reliability to .csv.

        display: bool, default=False
            Print output in the terminal.

        null: bool, default=True
            If True, estimate the chance level by permutation (Mantel-shuffle the
            features, compare two PCAs) and attach it to ``df.attrs["null"]`` so a later
            ``plot_omni(split_df, ...)`` draws the chance reference line + 95% CI band.

        null_reps: int, default=200
            Number of shuffled permutation draws used to estimate the chance level.

        path: str, default='results'
            The path to the output directory.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns
    -------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.

        .csv:
            If save=True, will save /results to a csv.

        printed results:
            If display=True, prints the output directly in the terminal.
    """

    if stratify is not None and group is not None:
        raise ValueError(
            "Pass either group= (per-level iteration) or stratify= (balanced sampling "
            "from each level across whole-dataset halves), not both."
        )

    drop_cols = [c for c in (group, cluster, stratify, groupby) if c is not None]
    df_t = df.drop(labels=drop_cols, axis=1) if drop_cols else df
    samples = df[group].unique() if group else ['fulldata']
    _check_rank(df_t)

    # R-homologue projection target. With groupby set, the components live in the
    # within-group-standardized space, so the common projection target must be too;
    # using the full data's per-group means here is a fixed symmetric transform (the
    # same for both halves), not a per-split estimate, so it introduces no leakage.
    rd_src = (group_standardize(df, groupby, feature_cols=list(df_t.columns))[list(df_t.columns)].values
              if groupby else df_t.values)
    boot_model = RHom(rd=copy.deepcopy(rd_src), n_comp=npc,
                      method=method, rotation=rotation, corr=corr)
    cv = pair_cv(group=group, cluster=cluster, stratify=stratify, n=boot, groupby=groupby)

    boot_engine = BootstrapEngine(
        estimator=boot_model,
        cv=cv,
        mode="splithalf",
        pro_cong=True,
        shuffle=False,
        subspace=subspace,
        progress=progress,
        progress_desc="Split-half bootstrap",
    )

    rows = []
    for sample in samples:
        print(f"Running Split-Half: {sample}")
        # Execute via the unified __call__ interface
        results = boot_engine(X=df, y=sample, group=group)

        meta = {group: sample} if group else {"Group": "fulldata"}
        row = _build_row(boot_model.n_comp, results[0], results[1],
                         sub_data=results[2] if subspace else None, metadata=meta)
        rows.append(row)

        if display:
            _display_stats(f"Split-Half Reliability for {sample}", row)

    split_df = pd.DataFrame(rows)

    # Chance reference: similarity of two PCAs of Mantel-shuffled (structure-free) data.
    # Attached to attrs so plot_omni(split_df, ...) draws it; splithalf itself has no plot.
    if null:
        split_df.attrs["null"] = _chance_reference(df_t, npc, method, rotation, corr,
                                                   subspace, null_reps, progress,
                                                   desc="Chance null (split-half)")
        split_df.attrs["null_reps"] = null_reps

    if save:
        _export_report(split_df, path, file_prefix, f"splithalf_{len(df_t.columns)}D_{npc}PC")

    return split_df


def splithalf_bypc(df=None, group=None, npc=None, method='svd', rotation='varimax', corr='pearson',
                   boot=1000, save=True, plot=True, display=False, null=True, null_reps=200, cluster=None,
                   stratify=None, groupby=None, subspace=False, progress=True,
                   path='results', file_prefix=randint(10000, 99999)):
    """
    Split-Half Reliability: By-Component
    ------------------------------------
    Bootstrapped split-half reliability with a per-component breakdown. For each
    sample (the whole dataset when ``group=None``, or each level of ``group``), an
    anchor PCA is fit on the sample's full data to define the canonical PC1..PC{npc}
    frame; every per-half PCA inside the bootstrap is Procrustes-aligned to that
    anchor (via ``rhom``'s ``anchor=`` parameter), so the column index k carries a
    consistent homologue identity across all replicates. The result is one row per
    (sample, comp) triple with rhm / phi (and optionally sub) CIs computed across
    bootstrap replicates.

    Same anchor-and-transpose architecture as ``dir_proj_bypc``, with the splithalf
    resampling protocol substituted for dir_proj's pairwise CV.

    Parameters
    ----------

        df: pd.Dataframe, default=None
            Decomposition columns plus the grouping / clustering / stratification
            columns if used.

        group: str, default=None
            Column heading for per-level iteration. With ``group=None`` the analysis
            runs on the whole dataset as one sample.

        cluster: str, default=None
            Optional level-2 / clustering column. Whole clusters are kept on one side
            of every bootstrap split.

        groupby: str, default=None
            Optional nuisance-grouping column for groupedPCA-style decomposition: each
            variable is z-scored within each level of this column before decomposing
            (per resample, on its own rows -- leakage-free), so components reflect
            within-group covariance rather than between-group differences. Independent
            of ``group`` / ``cluster`` / ``stratify``.

        stratify: str, default=None
            Optional stratification column for whole-dataset splithalf. When provided
            (and ``group`` is None), each bootstrap half is drawn proportionally from
            every level. Mutually exclusive with ``group``.

        subspace: bool, default=False
            If True, also report subspace similarity via principal angles. Subspace
            similarity is a whole-solution property, so per-replicate values are
            broadcast across the npc rows for that sample (same convention as
            ``omsamp_bypc`` / ``dir_proj_bypc``).

        corr: str, default="pearson"
            Correlation matrix for ``method='eigen'``. See ``splithalf`` for options.

        npc: int, default=None
            Number of components to extract per solution.

        rotation: str, default="varimax"
            Rotation applied to anchor and per-half PCAs.

        boot: int, default=1000
            Number of bootstrap halves to draw per sample.

        null: bool, default=True
            If True, estimate the chance level by permutation (Mantel-shuffle the
            features, compare two PCAs) and attach it to ``df.attrs["null"]`` + draw it on
            each component panel as a dashed reference line + shaded 95% CI band.

        null_reps: int, default=200
            Number of shuffled permutation draws used to estimate the chance level.

        save / plot / display / path / file_prefix:
            Same conventions as the other builtins.

    Returns
    -------
        pd.DataFrame:
            One row per (sample, comp) triple with rhm_*, phi_*, and optional sub_*
            summary columns. Anchor loadings for the first sample are attached via
            ``df.attrs["loadings"]`` so ``plot_bypc`` can render wordclouds without a
            disk round-trip.

        .csv:
            If save=True.

        .png:
            If plot=True, one ``plot_bypc`` figure per metric.
    """
    if stratify is not None and group is not None:
        raise ValueError(
            "Pass either group= (per-level iteration) or stratify= (balanced sampling "
            "from each level across whole-dataset halves), not both."
        )

    drop_cols = [c for c in (group, cluster, stratify, groupby) if c is not None]
    df_t = df.drop(labels=drop_cols, axis=1) if drop_cols else df
    samples = df[group].unique() if group else ['fulldata']
    _check_rank(df_t)

    df_input = df.copy()

    # R-homologue projection target, group-standardized (full data) under groupby so it
    # matches the within-group-standardized component space (see splithalf for rationale).
    feat_cols = list(df_t.columns)
    rd_src = (group_standardize(df_input, groupby, feature_cols=feat_cols)[feat_cols].values
              if groupby else df_input[feat_cols].values)

    cv = pair_cv(group=group, cluster=cluster, stratify=stratify, n=boot, groupby=groupby)

    rows = []
    anchor_loadings_by_sample = {}

    for sample in samples:
        print(f"Running By-Component Split-Half: {sample}")

        # Anchor PCA on this sample's full data (the whole dataset when group is None,
        # or the sample's rows when iterating per group level). With groupby set, the
        # anchor is itself a grouped solution: standardize within groupby first, then
        # decompose (basePCA's global scaling is then an identity).
        sample_src = df_input[df_input[group] == sample] if group else df_input
        if groupby:
            sample_data = group_standardize(sample_src, groupby, feature_cols=feat_cols)[feat_cols]
        else:
            sample_data = sample_src.drop(labels=drop_cols, axis=1, errors='ignore') if drop_cols else sample_src

        anchor_pca = basePCA(n_components=npc, rotation=rotation, method=method, corr=corr)
        anchor_pca.fit(sample_data)
        anchor_loadings_by_sample[sample] = anchor_pca.loadings.copy()

        boot_model = RHom(rd=copy.deepcopy(rd_src), bypc=True, n_comp=npc,
                          method=method, rotation=rotation, corr=corr,
                          anchor=anchor_pca.loadings.to_numpy())

        # mode='splithalf' selects cv.resample_pairs (random-half resampling). per_component=True
        # makes the engine transpose the score / phi accumulators into [npc × n_replicates]
        # lists at the end and collapse the per-direction subspace cosines to per-replicate
        # means -- so we don't have to do that work here. engine.shuffle=False because we've
        # already shuffled df_input once above.
        boot_engine = BootstrapEngine(
            estimator=boot_model,
            cv=cv,
            mode="splithalf",
            per_component=True,
            pro_cong=True,
            shuffle=False,
            subspace=subspace,
            progress=progress,
            progress_desc=f"Split-half bypc bootstrap ({sample})",
        )

        results = boot_engine(X=df_input, y=sample, group=group)
        # results layout with per_component=True (engine handles transpose + subspace collapse):
        #   results[0]: list-of-lists [npc × n_replicates]  -- |r| per component per replicate
        #   results[1]: list-of-lists [npc × n_replicates]  -- TCC per component per replicate
        #   results[2]: list of floats  [n_replicates]      -- per-replicate mean subspace cosine
        rhm_per_comp = results[0]
        phi_per_comp = results[1]
        sub_per_sample = results[2] if subspace else None

        for idx in range(npc):
            meta = ({group: sample} if group else {"Group": "fulldata"})
            meta["comp"] = idx + 1
            row = _build_row(boot_model.n_comp, rhm_per_comp[idx], phi_per_comp[idx],
                             sub_data=sub_per_sample, metadata=meta)
            rows.append(row)

            if display:
                _display_stats(f"Split-Half: {sample} - Component {idx + 1}", row)

    splithalf_bypc_df = pd.DataFrame(rows)

    # Chance reference (single floor shown on every component panel).
    null_ref = None
    if null:
        null_ref = _chance_reference(df_t, npc, method, rotation, corr, subspace,
                                     null_reps, progress, desc="Chance null (split-half bypc)")
        splithalf_bypc_df.attrs["null"] = null_ref
        splithalf_bypc_df.attrs["null_reps"] = null_reps

    if plot:
        # Attach anchor loadings so plot_bypc can render wordclouds. When group is set
        # each sample has its own anchor; we attach the first sample's by default --
        # users can pass their own via plot_bypc(..., loadings=anchor_loadings_by_sample[<g>]).
        first_sample = samples[0]
        splithalf_bypc_df.attrs["loadings"] = anchor_loadings_by_sample[first_sample]

        from ...visualization.rhomplots import plot_bypc
        setupanalysis(path, file_prefix, includetime=False)
        plt.close('all')

        metrics = ["rhm", "phi"] + (["sub"] if subspace else [])
        for name in metrics:
            fig = plot_bypc(splithalf_bypc_df, metric=name, null=null_ref)
            fig.savefig(
                os.path.join(path, f"{file_prefix}/{file_prefix}_splithalf_bypc_{len(df_t.columns)}D_{npc}PC_{name}.png"),
                bbox_inches="tight", dpi=150,
            )
            plt.show()
            plt.close(fig)

        # Persist the first sample's anchor loadings alongside the CSV
        loadings_path = os.path.join(
            path, f"{file_prefix}",
            f"{file_prefix}_loadings_{len(df_t.columns)}D_{npc}PC.csv",
        )
        anchor_loadings_by_sample[first_sample].to_csv(loadings_path)

    if save:
        _export_report(splithalf_bypc_df, path, file_prefix,
                       f"splithalf_bypc_{len(df_t.columns)}D_{npc}PC")

    return splithalf_bypc_df
