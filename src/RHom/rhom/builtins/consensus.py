"""
Consensus PCA: aggregated out-of-sample loadings.

Builds a stack of PCA solutions from bootstrap resamples or held-out CV folds,
Procrustes-aligns each to a shared omnibus anchor (via ``rhom.align_to_anchor``),
and returns the element-wise consensus loadings plus per-element 95% CIs.
"""
from ..._deps import pd, np, randint, plt

import os

from ...core.base_pca import basePCA
from ...preprocessing.preliminary import _check_rank
from ...preprocessing.data_utils import group_standardize
from ...io.save import setupanalysis

from ..metrics import RHom
from ..resampling import pair_cv

from ._reporting import _summary_stats, _export_report, _progress_wrap


def consensus_pca(df=None, group=None, folds=None, boot=None, npc=None,
                  method='svd', rotation='varimax', corr='pearson',
                  cluster=None, stratify=None, groupby=None, save=True, plot=True, display=False,
                  shuffle=False, keep_stack=False, progress=True, path='results',
                  file_prefix=randint(10000, 99999)):
    """
    Consensus PCA: aggregated "out-of-sample" loadings
    --------------------------------------------------
    Builds up a stack of PCA solutions from repeated resamples or held-out CV folds,
    Procrustes-aligns each to a common omnibus anchor (so column k carries a
    consistent homologue identity), and returns the element-wise consensus loadings
    plus per-element 95% bootstrap CIs. Implements the "average out-of-sample PCAs
    into one overarching set of dimensions" workflow.

    Three resampling modes, selected by which of ``boot`` / ``folds`` / ``group``
    are passed:

        * ``boot=B`` only -- bootstrap mode: B random subsamples (with replacement).
          Optional ``cluster=`` makes the bootstrap cluster-aware (whole clusters
          sampled with replacement); optional ``stratify=`` makes it stratified
          (within each level, rows are resampled with replacement). Mutually
          exclusive with ``folds`` / ``group``.
        * ``folds=K`` only -- random K-fold CV: fit on each fold's train portion.
        * ``group='colname'`` only -- leave-one-group-out: fit on N-1 groups each
          iteration.
        * ``group='colname'`` AND ``folds=K`` -- stratified K-fold by ``group``.

    Returns a long-format DataFrame with one row per (item, comp) holding the
    consensus mean, std, and 95% CI for each loading element. The wide-form
    consensus loadings (p × npc) and the omnibus anchor are attached via
    ``df.attrs["consensus"]`` and ``df.attrs["anchor"]`` for downstream plotting.

    Parameters
    ----------

        df: pd.Dataframe, default=None
            Decomposition columns plus the grouping / clustering / stratification
            columns if used.

        boot: int, default=None
            Number of bootstrap resamples. Mutually exclusive with ``folds`` /
            ``group``.

        folds: int, default=None
            Number of K-fold splits. Random partition when used alone; stratified
            by ``group`` when both are passed.

        group: str, default=None
            Column for LOGO splits when used alone, or stratifier when paired with
            ``folds``.

        cluster: str, default=None
            Optional level-2 / clustering column. In bootstrap mode, whole clusters
            are sampled with replacement. In CV modes, whole clusters are kept
            within a fold.

        groupby: str, default=None
            Optional nuisance-grouping column for groupedPCA-style decomposition: each
            variable is z-scored within each level of this column before decomposing
            (per resample, on its own rows -- leakage-free), so the consensus loadings
            describe a grouped solution. Independent of ``group`` / ``cluster`` /
            ``stratify``.

        stratify: str, default=None
            Optional stratification column for bootstrap mode (sample with replacement
            within each level). In CV modes use ``group`` as the stratifier instead.

        npc: int, default=None
            Number of components to extract per fit.

        rotation: str, default="varimax"
            Rotation applied to anchor and per-iteration PCAs.

        corr: str, default="pearson"
            Correlation matrix for ``method='eigen'``. See ``splithalf`` for options.

        shuffle: bool, default=False
            One-shot Mantel shuffle at the top, used for both the anchor fit and
            every resample (so the consensus describes the null realisation
            consistently). For a proper null distribution of consensus uncertainty
            you'd run this with multiple independent shuffles -- typically overkill
            but doable by looping over seeded calls.

        keep_stack: bool, default=False
            If True, attach the raw aligned-loadings stack (n_iter × p × npc) via
            ``result.attrs["stack"]`` for downstream custom aggregation (median,
            weighted, robust trimmed means, etc.).

        save / plot / display / path / file_prefix:
            Same conventions as the other builtins.

    Returns
    -------
        pd.DataFrame:
            Long form with one row per (item, comp) carrying ``loading_x`` (bootstrap
            mean), ``loading_se``, ``loading_LCI``, ``loading_UCI`` (percentile CIs).
            Schema matches RHom's metric convention so the same helpers that read
            ``rhm_x / phi_x / sub_x`` from other builtins can read ``loading_x`` here.
            Wide-form consensus and anchor attached via ``attrs``.

        .csv:
            If save=True.

        .png:
            If plot=True, a grid of (wordcloud, horizontal bar) panels via
            ``plot_consensus_pca``.
    """
    # Mode validation
    if boot is None and folds is None and group is None:
        raise ValueError(
            "consensus_pca requires at least one of boot= (bootstrap), folds= "
            "(k-fold), or group= (LOGO). All three were None."
        )
    if boot is not None and (folds is not None or group is not None):
        raise ValueError(
            "Pass boot= for bootstrap mode OR folds=/group= for CV mode, not both."
        )

    drop_cols = [c for c in (group, cluster, stratify, groupby) if c is not None]
    feat_cols = df.columns.drop(drop_cols) if drop_cols else df.columns
    _check_rank(df[feat_cols])

    # One-shot Mantel shuffle so anchor and resamples share the same null realisation.
    df_input = df.copy()
    if shuffle:
        from ...preprocessing.data_utils import fullmantel
        df_input[feat_cols] = fullmantel(df_input[feat_cols]).values

    # Omnibus anchor: defines the canonical PC1..PC{npc} frame. Under groupby it is a
    # within-group-standardized (grouped) pooled solution.
    anchor_src = (group_standardize(df_input, groupby, feature_cols=list(feat_cols))[list(feat_cols)]
                  if groupby else df_input[feat_cols])
    anchor_pca = basePCA(n_components=npc, rotation=rotation, method=method, corr=corr)
    anchor_pca.fit(anchor_src)
    anchor = anchor_pca.loadings.to_numpy()

    # Build the iterator over resamples / fold-train sets, delegating cluster /
    # stratify awareness to the corresponding pair_cv method.
    if boot is not None:
        cv = pair_cv(n=boot, cluster=cluster, stratify=stratify, groupby=groupby)
        mode = f"bootstrap (B={boot})"
        if cluster is not None:
            mode += f", cluster-aware on '{cluster}'"
        if stratify is not None:
            mode += f", stratified by '{stratify}'"

        def _iter_samples():
            # bootstrap_resamples doesn't standardize, so apply within-group
            # standardization here (per resample, on its own rows) when groupby is set.
            for sample in cv.bootstrap_resamples(df_input):
                if groupby:
                    yield group_standardize(sample, groupby, feature_cols=list(feat_cols))[feat_cols]
                else:
                    yield sample[feat_cols]

    else:
        # CV mode: identical three-way toggle as holdout_cv, so reuse its builder
        # rather than duplicating the dispatch and the mode strings.
        from .holdout import _make_holdout_cv
        cv, mode = _make_holdout_cv(group, folds, cluster, groupby=groupby)

        def _iter_samples():
            # holdout_split already applies per-fold within-group standardization via
            # _prep when groupby is set, so the train array is decomposition-ready.
            for train, _test, _label in cv.holdout_split(df_input):
                yield pd.DataFrame(train, columns=feat_cols)

    # Run the resamples: fit, align via rhom's shared anchor primitive, accumulate.
    # Total iterations: B for bootstrap mode, K (folds) or G (groups) for CV modes.
    if boot is not None:
        total_iter = boot
    elif folds is not None:
        total_iter = folds
    else:
        total_iter = int(df_input[group].nunique())

    print(f"Running Consensus PCA: {mode}")
    stack = []
    sample_iter = _progress_wrap(
        _iter_samples(),
        total=total_iter,
        desc=f"Consensus PCA ({mode})",
        enabled=progress,
    )
    for i, sample in enumerate(sample_iter):
        L = basePCA(n_components=npc, rotation=rotation,
                    method=method, corr=corr).fit(sample).loadings.values
        stack.append(RHom.align_to_anchor(L, anchor))
        if display and (i + 1) % max(1, len(stack) // 10) == 0:
            print(f"  iteration {i + 1}: fit + aligned")

    stack = np.stack(stack, axis=0)   # (n_iter, p, npc)
    n_iter = stack.shape[0]

    # Aggregate per element via the package's shared summary helper so the schema
    # matches the rest of RHom (loading_x / loading_se / loading_LCI / loading_UCI).
    items = list(feat_cols)
    rows = []
    for i, item in enumerate(items):
        for k in range(npc):
            stats = _summary_stats(stack[:, i, k])
            rows.append({
                "n_comp": f"{npc}PC",
                "item": item,
                "comp": k + 1,
                **{f"loading_{key}": val for key, val in stats.items()},
            })
    consensus_df = pd.DataFrame(rows)

    # Wide-form consensus + anchor for downstream plotting / inspection
    consensus_wide = pd.DataFrame(stack.mean(axis=0), index=items,
                                   columns=[f"PC{k+1}" for k in range(npc)])
    consensus_df.attrs["consensus"] = consensus_wide
    consensus_df.attrs["anchor"] = anchor_pca.loadings.copy()
    consensus_df.attrs["n_iter"] = n_iter

    if keep_stack:
        consensus_df.attrs["stack"] = stack

    if display:
        print(f"Consensus PCA: n_iter={n_iter}, npc={npc}, p={len(items)}, mode={mode}")

    if plot:
        from ...visualization.rhomplots import plot_consensus_pca
        setupanalysis(path, file_prefix, includetime=False)
        plt.close('all')

        fig = plot_consensus_pca(consensus_df,
                                  title=f"Consensus PCA ({mode}, n_iter={n_iter})")
        fig.savefig(
            os.path.join(path, f"{file_prefix}/{file_prefix}_consensus_{len(items)}D_{npc}PC.png"),
            bbox_inches="tight", dpi=150,
        )
        plt.show()
        plt.close(fig)

        # Persist the consensus loadings alongside the CSV
        loadings_path = os.path.join(
            path, f"{file_prefix}",
            f"{file_prefix}_consensus_loadings_{len(items)}D_{npc}PC.csv",
        )
        consensus_wide.to_csv(loadings_path)

    if save:
        _export_report(consensus_df, path, file_prefix,
                       f"consensus_{len(items)}D_{npc}PC")

    return consensus_df
