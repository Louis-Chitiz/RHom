"""
Omnibus-frame analyses.

``omni_sample`` is the aggregate omnibus-vs-sample reproducibility bootstrap;
``omsamp_bypc`` is the per-component breakdown of that comparison; ``omni_variance``
is the descriptive variance-attribution complement that asks how an omnibus
solution partitions variance within each group.
"""
from ..._deps import pd, np, randint, plt, StandardScaler

import os
import copy

from ...core.base_pca import basePCA
from ...preprocessing.preliminary import _check_rank
from ...preprocessing.data_utils import group_standardize
from ...io.save import setupanalysis
from ...visualization.wordclouds import save_wordclouds

from ..metrics import RHom
from ..resampling import pair_cv
from ..bootstrap import BootstrapEngine

from ._reporting import _build_row, _display_stats, _export_report, _chance_reference


def omni_sample(df=None, group=None, npc=None, method='svd', rotation="varimax", corr='pearson',
                boot=1000, save=True, display=False, plot=True, null=True, null_reps=200, cluster=None,
                groupby=None, subspace=False, progress=True, path='results', file_prefix=randint(10000, 99999)):
    """
    Omnibus-Sample Reproducibility
    ------------------------------
    This function conducts an omnibus-sample reproducibility analysis on your data.
    It randomly bootstrap reassigns halves of each level of an inputted grouping variable
    to be used in either a 'sample' or 'omnibus' subset. The 'sample' subsets generate
    components representative of that level of the grouping variable, while the 'omnibus'
    subsets are aggregated with other groups to produce 'common' components. The analysis
    assesses the component similarity of the orthogonal aggregated set relative to each sample.
    It computes component similarity with:
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
            of ``group``. Setting ``groupby=group`` is valid and measures the
            reproducibility of the grouped (within-``group`` standardized) solution --
            the right estimand when that is the model you report. It does not, however,
            test whether *raw* between-group differences threaten blendability (the
            other way omnibus-sample is read); use ``groupby=None`` for that, or compare
            the two.

        subspace: bool, default=False
            If True, also report subspace similarity via principal angles between the two
            loading subspaces (sub_* columns; rotation- and order-invariant, in [0, 1]).

        corr: str, default="pearson"
            Which correlation matrix to decompose under `method='eigen'`: "pearson",
            "spearman" (rank correlation, ordinal-friendly), or "polychoric" (latent
            correlation behind ordinal items via Olsson 1979 MLE; meaningful only for
            genuinely ordinal data and noticeably slower). Ignored under `method='svd'`,
            which is Pearson-only; pass `method='eigen'` to switch correlation type.

        groupby: str, default=None
            If used, the omnibus sets using groupwise standardization.

        npc: int, default=None
            Number of components to extract per solution.

        rotation: str, default="varimax"
            Rotation method to be performed on omnibus set. "none" for no rotation.

        boot: int, default=1000
            Number of bootstrap samples to generate 95% confidence intervals.

        save: bool, default=True
            Save outputted omnibus-sample reliability to .csv.

        display: bool, default=False
            Print output in the terminal.

        null: bool, default=True
            If True, estimate the chance level by permutation (Mantel-shuffle the
            features, compare two PCAs) and attach it to ``df.attrs["null"]`` + draw it on
            the figure as a dashed reference line + shaded 95% CI band.

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

    drop_cols = [c for c in (group, cluster, groupby) if c is not None]
    samples = df[group].unique()
    df_t = df.drop(labels=drop_cols, axis=1)
    _check_rank(df_t)

    # R-homologue projection target: group-standardized (full data) under groupby so it
    # matches the within-group component space (see splithalf for rationale).
    rd_src = (group_standardize(df, groupby, feature_cols=list(df_t.columns))[list(df_t.columns)].values
              if groupby else df_t.values)
    boot_model = RHom(rd=copy.deepcopy(rd_src), n_comp=npc,
                      method=method, rotation=rotation, corr=corr)
    cv = pair_cv(omnibus=True, group=group, cluster=cluster, n=boot, groupby=groupby)

    # Initialize engine for omnibus resampling profile
    boot_engine = BootstrapEngine(
        estimator=boot_model,
        cv=cv,
        mode="omnibus",
        pro_cong=True,
        shuffle=False,
        subspace=subspace,
        progress=progress,
        progress_desc="Omnibus-sample bootstrap",
    )

    rows = []
    total_rhm, total_phi, total_sub = [], [], []

    for sample in samples:
        print(f"Running Omnibus x {sample}")
        results = boot_engine(X=df, y=sample, group=group)

        total_rhm.extend(results[0])
        total_phi.extend(results[1])
        if subspace:
            total_sub.extend(results[2])

        row = _build_row(boot_model.n_comp, results[0], results[1],
                         sub_data=results[2] if subspace else None, metadata={group: sample})
        rows.append(row)

        if display:
            _display_stats(f'Omnibus x {sample}', row)

    # Append Global Summary Row
    total_row = _build_row(boot_model.n_comp, total_rhm, total_phi,
                           sub_data=total_sub if subspace else None, metadata={group: "Total"})
    rows.append(total_row)

    if display:
        _display_stats("Overall Omnibus Summary", total_row)

    omsamp_df = pd.DataFrame(rows)

    # Chance reference: similarity of two PCAs of Mantel-shuffled (structure-free) data.
    null_ref = None
    if null:
        null_ref = _chance_reference(df_t, npc, method, rotation, corr, subspace,
                                     null_reps, progress, desc="Chance null (omnibus)",
                                     groupby_labels=(df[groupby].values if groupby else None),
                                     group_labels=df[group].values)
        omsamp_df.attrs["null"] = null_ref
        omsamp_df.attrs["null_reps"] = null_reps

    if plot:
        from ...visualization.rhomplots import plot_omni
        setupanalysis(path, file_prefix, includetime=False)

        plt.close('all')
        metrics = ["rhm", "phi"] + (["sub"] if subspace else [])
        for metric in metrics:
            fig = plot_omni(omsamp_df, metric=metric, null=null_ref,
                            title=f"Omnibus-Sample Reproducibility: {metric.upper()}", n_vars=len(df_t.columns))
            fig.savefig(os.path.join(path, f"{file_prefix}/{file_prefix}_omsamp_{len(df_t.columns)}D_{npc}PC_{metric}.png"), bbox_inches="tight", dpi=150)
            plt.show()
            plt.close(fig)

    if save:
        _export_report(omsamp_df, path, file_prefix, f"omsamp_{len(df_t.columns)}D_{npc}PC")

    return omsamp_df


def omni_variance(df=None, group=None, npc=None, method='svd', rotation='varimax',
                  corr='pearson', cluster=None, groupby=None, save=True, plot=True, display=False,
                  path='results', file_prefix=randint(10000, 99999)):
    """
    Omnibus Variance Attribution
    ----------------------------
    Fits one omnibus PCA on the pooled data and reports, for each group, the share
    of the group's standardised variance captured by each omnibus component. This
    answers "given these pooled-data components, how do they partition variance
    within each group?" -- a descriptive complement to the reproducibility builtins,
    which instead ask "do these components agree across groups?".

    The npc bars stacked together (Σₖ varₘ(k)) give the total variance captured by
    the omnibus solution in group g. See the Notes section for the formal definition
    and the random-subspace chance baseline.

    Parameters
    ----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        cluster: str, default=None
            Optional level-2 / clustering column. Stripped before decomposition so a
            cluster ID does not contaminate the PCA, but has no effect on the variance
            attribution itself (which is deterministic and does not resample).

        groupby: str, default=None
            Optional nuisance-grouping column. When set, the omnibus PCA is a grouped
            (within-``groupby`` standardized) solution; the per-group variance
            attribution is otherwise unchanged. Independent of ``group``. With
            ``groupby=group`` the figure reports how much within-group variance the
            grouped components capture -- appropriate when groupedPCA is your model.

        corr: str, default="pearson"
            Correlation matrix used for the omnibus fit under ``method='eigen'``:
            "pearson", "spearman", or "polychoric". Ignored under ``method='svd'``.

        npc: int, default=None
            Number of components to extract for the omnibus PCA.

        rotation: str, default="varimax"
            Rotation applied to the omnibus loadings. "none" for no rotation.

        save: bool, default=True
            Save the (group, comp, var_pct) table to .csv.

        plot: bool, default=True
            Render the stacked-bar variance attribution figure.

        display: bool, default=False
            Print each group's per-component variance share in the terminal.

        path: str, default='results'
            The path to the output directory.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files.

    Returns
    -------
        pd.DataFrame:
            One row per (group, comp) with columns ``n_comp``, ``group`` (named after
            your grouping variable), ``comp``, ``var_pct``.

        .csv:
            If save=True.

        .png:
            If plot=True, a stacked-bar figure with one bar per group and npc
            segments stacked by component contribution.

    Notes
    -----
    Let

    * $L \\in \\mathbb{R}^{p \\times \\text{npc}}$ represent the omnibus loadings (basePCA output)
    * $\\ell_k = L[:, k] / \\Vert{}L[:, k]\\Vert{}$ represent the unit-normalised $k$-th column
    * $\\tilde{X}_g \\in \\mathbb{R}^{n_g \\times p}$ represent group $g$'s data after column-wise standardisation
    * $R_g = \\tilde{X}_g^T \\tilde{X}_g / n_g$ represent the within-group correlation matrix
    * $p$ represent the number of decomposed features

    Then component $k$'s variance share in group $g$ is

    $$\\text{var}^g(k) = \\frac{\\ell_k^T \\cdot R_g \\cdot \\ell_k}{p} \\times 100\\%$$

    which equals, equivalently, the projected-score formulation

    $$\\text{var}^g(k) = \\frac{\\Vert{}\\tilde{X}_g \\cdot \\ell_k\\Vert{}^2}{n_g \\cdot p} \\times 100\\%$$

    used inside the implementation. Summing over $k$ gives the total variance that
    the omnibus solution captures in group $g$ ($\\le 100\\%$; equal to $100\\%$ only when
    $\\text{npc} = p$).

    A random $\\text{npc}$-dimensional subspace on standardised isotropic data captures, in
    expectation, $\\frac{\\text{npc}}{p} \\times 100\\%$ of the within-group variance, which the
    `plot_omni_variance` figure marks with a dashed horizontal line. Because
    variance is a *quadratic* projection, this baseline is the *square* of the
    canonical-correlation baseline $\\sqrt{\\frac{\\text{npc}}{p}}$ used for the
    reproducibility metrics (rhm, phi, sub).

    .. math::

        \\mathrm{var}_g(k) = \\frac{\\ell_k^\\top R_g \\,\\ell_k}{p} \\times 100\\%,
        \\qquad
        \\text{chance} \\approx \\frac{\\mathrm{npc}}{p} \\times 100\\%.
    """

    drop_cols = [c for c in (group, cluster, groupby) if c is not None]
    feat_cols = df.columns.drop(drop_cols)
    _check_rank(df[feat_cols])

    # Omnibus PCA on pooled data defines the reference component set. Under groupby it is
    # a within-group-standardized (grouped) pooled solution; the per-group variance
    # attribution below is unchanged (it standardizes within each attribution group).
    omni_src = (group_standardize(df, groupby, feature_cols=list(feat_cols))[list(feat_cols)]
                if groupby else df[feat_cols])
    omni = basePCA(n_components=npc, rotation=rotation, method=method, corr=corr)
    omni.fit(omni_src)
    L = omni.loadings.values

    # basePCA stores loadings as eigvec * sqrt(eigval); for variance attribution we
    # want unit-norm directions, so divide each column by its norm.
    norms = np.linalg.norm(L, axis=0)
    L_unit = L / norms

    p = len(feat_cols)
    samples = df[group].unique()

    rows = []
    for g in samples:
        X_g = df.loc[df[group] == g, feat_cols]
        X_g_std = StandardScaler().fit_transform(X_g)
        scores = X_g_std @ L_unit                                # (n_g, npc)
        var_per_comp = (scores ** 2).sum(axis=0) / len(X_g_std)  # variance of each PC's scores
        var_pct = var_per_comp / p * 100                         # share of group's total variance

        for k in range(npc):
            rows.append({
                "n_comp": f"{npc}PC",
                group: g,
                "comp": k + 1,
                "var_pct": float(var_pct[k]),
            })

        if display:
            shares = ", ".join(f"PC{k+1}={var_pct[k]:.1f}%" for k in range(npc))
            total = float(var_pct.sum())
            print(f"{g}: total={total:.1f}% | {shares}")

    omni_var_df = pd.DataFrame(rows)

    if plot:
        from ...visualization.rhomplots import plot_omni_variance
        setupanalysis(path, file_prefix, includetime=False)
        plt.close('all')

        fig = plot_omni_variance(omni_var_df, group=group, n_vars=p)
        fig.savefig(
            os.path.join(path, f"{file_prefix}/{file_prefix}_omni_var_{p}D_{npc}PC.png"),
            bbox_inches="tight", dpi=150,
        )
        plt.show()
        plt.close(fig)

    if save:
        _export_report(omni_var_df, path, file_prefix, f"omni_var_{p}D_{npc}PC")

    return omni_var_df


def omsamp_bypc(df=None, group=None, npc=None, method='svd', rotation="varimax", corr='pearson',
         folds=5, save=True, plot=True, display=False, null=True, null_reps=200, cluster=None,
         groupby=None, subspace=False, progress=True, path='results', file_prefix=randint(10000, 99999)):

    """
    Omnibus-Sample Reproducibility: By-Component
    ------------------------------
    This function conducts an omnibus-sample reproducibility analysis on your data,
    modified to assess the correspondence between each component of an omnibus solution and its
    corresponding components in each subset. It randomly reassigns halves of each level
    of an inputted grouping variable to be stably used in either a 'sample' or 'omnibus' subset.
    The 'sample' subsets are folded to generate cross-validated components representative of that level of
    the grouping variable, while the 'omnibus' subset is aggregated with other groups to produce 'common' components.
    The analysis assesses the component similarity of the orthogonal aggregated set relative to each sample.
    It computes component similarity with:
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
            of ``group``. Setting ``groupby=group`` is valid and measures the
            reproducibility of the grouped (within-``group`` standardized) solution --
            the right estimand when that is the model you report. It does not, however,
            test whether *raw* between-group differences threaten blendability (the
            other way omnibus-sample is read); use ``groupby=None`` for that, or compare
            the two.

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
            Rotation method to be performed on omnibus set. "none" for no rotation.

        folds: int, default=5
            Number of folds to use for cross-validation.

        save: bool, default=True
            Save outputted omnibus-sample reliability to .csv.

        plot: bool, default=True
            Save wordclouds, a Scree plot, and .csv files for the specified omnibus set.

        display: bool, default=False
            Print output in the terminal.

        null: bool, default=True
            If True, estimate the chance level by permutation (Mantel-shuffle the
            features, compare two PCAs) and attach it to ``df.attrs["null"]`` + draw it on
            the figure as a dashed reference line + shaded 95% CI band.

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

        RHom PCA results:
            If plot=True, will save the results, including wordclouds and Scree plot, for the omnibus set in a RHom folder.

        printed results:
            If display=True, prints the output directly in the terminal.
    """

    drop_cols = [c for c in (group, cluster, groupby) if c is not None]
    df_t = df.drop(labels=drop_cols, axis=1)
    _check_rank(df_t)
    # R-homologue projection target: group-standardized (full data) under groupby so it
    # matches the within-group component space (see splithalf for rationale).
    rd_src = (group_standardize(df, groupby, feature_cols=list(df_t.columns))[list(df_t.columns)].values
              if groupby else df_t.values)
    boot_model = RHom(rd=copy.deepcopy(rd_src), bypc=True, n_comp=npc,
                      method=method, rotation=rotation, corr=corr)
    cv = pair_cv(boot=True, group=group, cluster=cluster, k=folds, groupby=groupby)

    nval = (df[group].value_counts().min()) / 2
    maindict = cv.omni_prep(df=df, subrows=nval)
    samples = df[group].unique()

    # Initialize engine tailored for granular by-component splits. The two flags are
    # now independent: mode='asym' selects the asymmetric (X-fixed/y-folded) resampling
    # via cv.asym_split, and per_component=True triggers the per-component transpose at
    # the end. Previously a single bypc=True flag did both jobs.
    boot_engine = BootstrapEngine(
        estimator=boot_model,
        cv=cv,
        mode="asym",
        per_component=True,
        pro_cong=True,
        shuffle=False,
        subspace=subspace,
        progress=progress,
        progress_desc="Omnibus-sample bypc CV",
    )

    rows = []
    for sample in samples:
        print(f"Running Component Breakdown: omnibus x {sample}")
        # Returns [complist, philist(, sublist)] -> [n_components, n_fold_combinations].
        # sublist (if present) is sample-level subspace similarity per fold combination, not
        # per-component, so it is reused across this sample's component rows.
        comps = boot_engine(X=maindict['omnibus'], y=maindict[sample], group=group)
        sub = comps[2] if subspace else None

        for idx in range(npc):
            meta = {group: sample, 'comp': idx + 1}
            row = _build_row(boot_model.n_comp, comps[0][idx], comps[1][idx],
                             sub_data=sub, metadata=meta)
            rows.append(row)

            if display:
                _display_stats(f"Omnibus x {sample} - Component {idx + 1}", row)

    stats_bypc = pd.DataFrame(rows)

    # Chance reference (single floor shown on every component panel; under permutation
    # the per-component chance is the same structure-free level).
    null_ref = None
    if null:
        null_ref = _chance_reference(df_t, npc, method, rotation, corr, subspace,
                                     null_reps, progress, desc="Chance null (omnibus bypc)",
                                     groupby_labels=(df[groupby].values if groupby else None),
                                     group_labels=df[group].values, bypc=True)
        stats_bypc.attrs["null"] = null_ref
        stats_bypc.attrs["null_reps"] = null_reps

    if plot:
        # Fit basePCA on the exact omnibus half drawn in this call, so the wordclouds
        # match the loadings that underlie the per-component similarity scores above.
        om_model = basePCA(n_components=npc, rotation=rotation, method=method)
        om_model.fit(maindict['omnibus'])

        cloud_dir = setupanalysis(os.path.join(path, str(file_prefix)), "bypc_wordclouds", includetime=False)
        save_wordclouds(om_model.loadings, path=str(cloud_dir))

        # Persist the loadings alongside the stats CSV and attach them to the returned
        # frame so plot_bypc can render the wordclouds without a save/reload round-trip.
        loadings_path = os.path.join(
            path, f"{file_prefix}",
            f"{file_prefix}_loadings_{len(df_t.columns)}D_{npc}PC.csv",
        )
        om_model.loadings.to_csv(loadings_path)
        stats_bypc.attrs["loadings"] = om_model.loadings.copy()

        from ...visualization.rhomplots import plot_bypc
        plt.close('all')
        metrics = ["rhm", "phi"] + (["sub"] if subspace else [])
        for metric in metrics:
            fig = plot_bypc(stats_bypc, metric=metric, null=null_ref)
            fig.savefig(os.path.join(path, f"{file_prefix}/{file_prefix}_bypc_{len(df_t.columns)}D_{npc}PC_{metric}.png"), bbox_inches="tight", dpi=150)
            plt.show()
            plt.close(fig)

    if save:
        _export_report(stats_bypc, path, file_prefix, f"bypc_{len(df_t.columns)}D_{npc}PC")

    return stats_bypc
