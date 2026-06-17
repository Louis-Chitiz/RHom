"""
Held-out cross-validation reproducibility.

``holdout_cv`` is the aggregate analysis with a three-way mode toggle: LOGO, random
K-fold, or stratified K-fold by a grouping variable. ``holdout_bypc`` is its
per-component breakdown, anchored against a pooled-data PCA so component identity
is stable across folds (same architecture as ``dir_proj_bypc``).

Both reuse ``rhom`` unchanged on train / test pairs from ``pair_cv.holdout_split``.
For the two K-fold modes, an optional ``boot=B`` repeats the random partition B
times (repeated K-fold cross-validation) so the cross-validated reproducibility
estimate comes with a bootstrap confidence interval over partitions rather than
resting on a single arbitrary split. Bootstrapping does not apply to LOGO, whose
train/test split is deterministic.

The single-partition and repeated-K-fold iteration patterns are unified in one
private generator, ``_holdout_blocks``, so both public functions share a single
row-building loop instead of carrying duplicate boot / non-boot branches.
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

from ._reporting import _build_row, _display_stats, _export_report, _progress_wrap


def _make_holdout_cv(group, folds, cluster, groupby=None):
    """
    Build the ``pair_cv`` splitter and a human-readable mode label for the three
    held-out modes, selected by which of ``group`` / ``folds`` are passed:

        * ``group`` and ``folds``  -> stratified K-fold by ``group``
        * ``group`` only           -> leave-one-group-out
        * ``folds`` only           -> random K-fold (cluster-aware via ``cluster``)

    ``groupby`` (optional) is threaded through unchanged so each fold is
    standardized within that nuisance grouping (groupedPCA semantics per fold).
    """
    if group is not None and folds is not None:
        cv = pair_cv(k=folds, cluster=cluster, stratify=group, stratified_kfold=True, groupby=groupby)
        mode = f"{folds}-fold stratified by '{group}'"
    elif group is not None:
        cv = pair_cv(group=group, cluster=cluster, groupby=groupby)
        mode = f"leave-one-group-out by '{group}'"
    else:
        cv = pair_cv(k=folds, cluster=cluster, groupby=groupby)
        mode = f"{folds}-fold"
    return cv, mode


def _score_fold(boot_model, train, test, subspace):
    """
    Fit ``boot_model`` on one train/test fold pair and return ``(rhm, phi, sub)``.

    The shapes follow the estimator's ``bypc`` flag: with ``bypc=False`` (aggregate)
    ``rhm`` and ``phi`` are floats; with ``bypc=True`` (per-component, anchored) they
    are length-``npc`` lists indexed by component. Subspace similarity is a whole-
    solution property, so it is always collapsed to a single float per fold
    (broadcast across components by the caller in the bypc case).
    """
    boot_model.fit(train, test)
    preds = boot_model.predict()
    corrs = np.corrcoef(preds[0], preds[1], rowvar=False)

    rhm = boot_model.hom_pairs(corrs)
    phi = boot_model.pro_cong()

    sub = None
    if subspace:
        s = boot_model.subspace_sim()
        sub = float(np.mean(s)) if isinstance(s, (list, tuple, np.ndarray)) else float(s)

    return rhm, phi, sub


def _holdout_blocks(cv, df_input, boot_model, subspace, boot, total_folds, mode,
                    progress, tag=""):
    """
    Yield labeled blocks of per-fold scores, unifying the single-partition and
    repeated-K-fold (boot) iteration patterns so callers need only one loop.

    Each yielded item is ``(label, rhm_block, phi_block, sub_block)`` where the
    blocks are lists of per-fold scores (a score is a float for the aggregate
    estimator, or a length-``npc`` per-component list when ``boot_model.bypc`` is
    set -- ``_score_fold`` handles that transparently):

        * ``boot is None`` -> one item per fold; each block holds that single fold's
          score, so ``label`` is the fold/group name and the row gets a deterministic
          zero-width CI.
        * ``boot=B``        -> one item per repeat; each block holds that repeat's K
          fold scores, so ``label`` is ``"repeat{b}"`` and the row carries a
          within-repeat CI across the K folds.

    ``sub_block`` is None when ``subspace`` is False. The progress bar wraps the
    outer iterator -- folds without boot, repeats with boot -- so its denominator
    matches what the user is waiting on.
    """
    if boot is None:
        fold_iter = _progress_wrap(
            cv.holdout_split(df_input), total=total_folds,
            desc=f"Held-out CV{tag} ({mode})", enabled=progress,
        )
        for train, test, label in fold_iter:
            rhm, phi, sub = _score_fold(boot_model, train, test, subspace)
            yield label, [rhm], [phi], ([sub] if subspace else None)
        return

    repeat_iter = _progress_wrap(
        range(boot), total=boot,
        desc=f"Repeated K-fold{tag} ({mode})", enabled=progress,
    )
    for b in repeat_iter:
        # A fresh call to holdout_split reshuffles the partition (cluster-aware and
        # stratified as configured), so each repeat is an independent split.
        rhm_block, phi_block, sub_block = [], [], []
        for train, test, _label in cv.holdout_split(df_input):
            rhm, phi, sub = _score_fold(boot_model, train, test, subspace)
            rhm_block.append(rhm)
            phi_block.append(phi)
            if subspace:
                sub_block.append(sub)
        yield f"repeat{b + 1}", rhm_block, phi_block, (sub_block if subspace else None)


def holdout_cv(df=None, group=None, folds=None, boot=None, npc=None, method='svd',
               rotation='varimax', corr='pearson', cluster=None, groupby=None, save=True, plot=True,
               display=False, shuffle=False, subspace=False, progress=True,
               path='results', file_prefix=randint(10000, 99999)):
    """
    Held-Out Cross-Validation
    -------------------------
    Cross-validated component reproducibility: in each fold the PCA is fit on the
    training rows and the held-out rows independently, and the two solutions are
    compared via rhm / phi (and optionally sub). Three modes, selected by which of
    ``group`` / ``folds`` are passed:

        * ``group='colname'`` only -- leave-one-group-out: each level of ``colname``
          is held out as the test set once.
        * ``folds=K`` only -- random K-fold partition (cluster-aware when
          ``cluster=`` is set, so whole clusters land in one fold).
        * ``group='colname'`` AND ``folds=K`` -- stratified K-fold: each fold pulls
          proportional rows from every level of ``colname`` (cluster-aware within
          each stratum when ``cluster=`` is also set).

    By default each fold returns a point estimate (no bootstrap inside a fold) and
    the final "summary" row holds the mean and across-fold CI of each metric.

    Bootstrapped (repeated) K-fold
    ------------------------------
    Passing ``boot=B`` repeats the random partition B times -- repeated K-fold
    cross-validation. Each repeat draws a fresh random K-fold (or stratified
    K-fold) split and yields its own cross-validated estimate (the mean across that
    repeat's K folds); the headline "summary" row then reports the mean and 95%
    bootstrap CI of those B per-repeat estimates, so the CI reflects how stable the
    estimate is across partitions rather than resting on one arbitrary split. The
    returned frame additionally carries one ``fold='repeat{b}'`` row per repeat (its
    CV estimate with a within-repeat CI across the K folds); the saved figure shows
    only the summary. ``boot`` is only valid in the two K-fold modes -- LOGO has a
    deterministic split, so there is nothing to resample.

    Parameters
    ----------

        df: pd.Dataframe, default=None
            Decomposition columns plus the grouping / clustering columns if used.

        group: str, default=None
            Column name driving leave-one-group-out splits when used alone, or the
            stratifier when paired with ``folds=K``.

        folds: int, default=None
            Number of K-fold splits. Random partition when used alone; stratified by
            ``group`` when both are passed.

        boot: int, default=None
            Number of times to repeat the random partition (repeated K-fold CV). When
            None (default) a single partition is used. Only valid with ``folds=K``
            (random or stratified K-fold); raises if combined with a LOGO-only call.

        cluster: str, default=None
            Optional level-2 / clustering column. With ``folds=K`` (random or
            stratified), whole clusters are kept together within a fold (requires at
            least K distinct clusters; in stratified mode, K within each stratum).
            With ``group=...`` only, cluster is irrelevant since groups already define
            the partition.

        groupby: str, default=None
            Optional nuisance-grouping column for groupedPCA-style decomposition: each
            variable is z-scored within each level of this column before decomposing
            (per fold, on its own rows -- leakage-free), so components reflect
            within-group covariance rather than between-group differences. Independent
            of ``group`` / ``cluster``.

        subspace: bool, default=False
            If True, also report subspace similarity via principal angles.

        corr: str, default="pearson"
            Correlation matrix for ``method='eigen'``: "pearson", "spearman", or
            "polychoric". Ignored under ``method='svd'``.

        npc: int, default=None
            Number of components to extract per fold.

        rotation: str, default="varimax"
            Rotation applied to each fold's PCA.

        shuffle: bool, default=False
            If True, Mantel-shuffle the feature columns first (destroys cross-variable
            structure while preserving marginals) to produce a noise-floor null run.

        save / plot / display / path / file_prefix:
            Same conventions as the other builtins.

    Returns
    -------
        pd.DataFrame:
            Without ``boot``: one row per fold with ``rhm_x`` / ``phi_x`` (and
            ``sub_x`` if requested) point estimates, plus a final ``fold='summary'``
            row holding the mean and across-fold CI for each metric. With ``boot``:
            one ``fold='repeat{b}'`` row per repeat plus the ``fold='summary'`` row
            holding the bootstrap mean and 95% CI over the per-repeat estimates.

        .csv:
            If save=True.

        .png:
            If plot=True, one horizontal-bar figure per metric via ``plot_omni``
            (all rows without ``boot``; the summary row only with ``boot``).
    """
    if group is None and folds is None:
        raise ValueError(
            "holdout_cv requires at least one of group= (leave-one-group-out) or "
            "folds= (k-fold). Got both None."
        )
    if boot is not None and folds is None:
        raise ValueError(
            "boot= repeats the random partition (repeated K-fold CV), so it requires "
            "folds= (random or stratified K-fold). Leave-one-group-out (group= alone) "
            "has a deterministic split with nothing to resample."
        )

    drop_cols = [c for c in (group, cluster, groupby) if c is not None]
    feat_cols = df.columns.drop(drop_cols) if drop_cols else df.columns
    _check_rank(df[feat_cols])

    # Optional Mantel shuffle for the null baseline: column-wise independent
    # permutation destroys cross-variable structure while preserving marginals.
    # Pass only the feature columns so fullmantel doesn't accidentally treat a
    # numeric cluster ID as a feature; reassign by .values to overwrite in place.
    df_input = df.copy()
    if shuffle:
        from ...preprocessing.data_utils import fullmantel
        df_input[feat_cols] = fullmantel(df_input[feat_cols]).values

    cv, mode = _make_holdout_cv(group, folds, cluster, groupby=groupby)
    # R-homologue projection target: group-standardized (full data) under groupby so it
    # matches the within-group component space (see splithalf for rationale).
    rd_src = (group_standardize(df_input, groupby, feature_cols=list(feat_cols))[list(feat_cols)].values
              if groupby else df_input[feat_cols].values)
    boot_model = RHom(rd=copy.deepcopy(rd_src), n_comp=npc,
                      method=method, rotation=rotation, corr=corr)

    # Total folds: K for k-fold modes (random or stratified), G for LOGO.
    total_folds = folds if folds is not None else int(df_input[group].nunique())

    print(f"Running {'Bootstrapped ' if boot else ''}Held-Out CV "
          f"({mode}{f', B={boot}' if boot else ''})")

    rows = []
    # Summary distribution: pooled fold scores without boot, per-repeat CV means with.
    sum_rhm, sum_phi, sum_sub = [], [], []
    for label, rhm_block, phi_block, sub_block in _holdout_blocks(
            cv, df_input, boot_model, subspace, boot, total_folds, mode, progress):
        # One row per fold (boot None) or per repeat (boot=B). A single-value block
        # collapses to a zero-width CI; a K-value block carries a within-repeat CI.
        rows.append(_build_row(boot_model.n_comp, rhm_block, phi_block,
                               sub_data=sub_block, metadata={"fold": label}))

        if boot is None:
            sum_rhm += rhm_block
            sum_phi += phi_block
            if subspace:
                sum_sub += sub_block
        else:
            sum_rhm.append(float(np.mean(rhm_block)))
            sum_phi.append(float(np.mean(phi_block)))
            if subspace:
                sum_sub.append(float(np.mean(sub_block)))

        if display:
            extras = f", sub={np.mean(sub_block):.3g}" if subspace else ""
            print(f"  {label}: rhm={np.mean(rhm_block):.3g}, "
                  f"phi={np.mean(phi_block):.3g}{extras}")

    summary_row = _build_row(boot_model.n_comp, sum_rhm, sum_phi,
                             sub_data=sum_sub if subspace else None,
                             metadata={"fold": "summary"})
    rows.append(summary_row)

    if display:
        _display_stats(f"Held-Out CV summary ({mode})", summary_row)

    holdout_df = pd.DataFrame(rows)

    if plot:
        from ...visualization.rhomplots import plot_omni
        setupanalysis(path, file_prefix, includetime=False)
        plt.close('all')

        # Without boot, show every fold plus the summary. With boot the per-repeat
        # rows would crowd the axis (and the summary is the headline), so plot only
        # the summary row.
        plot_source = holdout_df if boot is None else holdout_df[holdout_df["fold"] == "summary"]

        metrics = ["rhm", "phi"] + (["sub"] if subspace else [])
        for name in metrics:
            # plot_omni's "Total" guard drops the omni_sample summary row; here we want
            # the summary row visible (it's the headline number), so we pass group='fold'
            # explicitly and rely on the label being 'summary' (not 'Total').
            fig = plot_omni(plot_source, group="fold", metric=name,
                            n_vars=len(feat_cols),
                            title=f"Held-Out CV ({mode}): {name.upper()}")
            fig.savefig(
                os.path.join(path, f"{file_prefix}/{file_prefix}_holdout_cv_{len(feat_cols)}D_{npc}PC_{name}.png"),
                bbox_inches="tight", dpi=150,
            )
            plt.show()
            plt.close(fig)

    if save:
        _export_report(holdout_df, path, file_prefix,
                       f"holdout_cv_{len(feat_cols)}D_{npc}PC")

    return holdout_df


def holdout_bypc(df=None, group=None, folds=None, boot=None, npc=None, method='svd',
                 rotation='varimax', corr='pearson', cluster=None, groupby=None, save=True, plot=True,
                 display=False, shuffle=False, subspace=False, progress=True,
                 path='results', file_prefix=randint(10000, 99999)):
    """
    Held-Out Cross-Validation: By-Component
    ---------------------------------------
    The per-component breakdown of ``holdout_cv``. A pooled-data PCA on the full
    dataset defines a canonical PC1..PC{npc} frame; every per-fold PCA (train and
    test side alike) is Procrustes-aligned to that anchor before scoring, so the
    column index k refers to the same homologue across all folds. This answers
    "which individual components survive cross-validation?" rather than
    ``holdout_cv``'s whole-solution average.

    Same three modes as ``holdout_cv`` (LOGO / random K-fold / stratified K-fold),
    and the same optional ``boot=B`` repeated-K-fold bootstrap (K-fold modes only).

    Reports component similarity with:
        1) Loading similarity (Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (R-homologue: Mulholland et al., 2023; See also Everett, 1983)

    Parameters
    ----------

        df: pd.Dataframe, default=None
            Decomposition columns plus the grouping / clustering columns if used.

        group / folds / boot / cluster / groupby / subspace / corr / npc / rotation / shuffle:
            Same meaning as in ``holdout_cv``.

        save / plot / display / path / file_prefix:
            Same conventions as the other builtins.

    Returns
    -------
        pd.DataFrame:
            Without ``boot``: one row per (fold, comp) point estimate plus, per
            component, a ``fold='summary'`` row holding the across-fold mean and CI.
            With ``boot``: one ``fold='summary'`` row per component holding the
            bootstrap mean and 95% CI over the per-repeat per-component estimates.
            Anchor loadings are attached via ``df.attrs["loadings"]`` so ``plot_bypc``
            can render the wordclouds without a disk round-trip.

        .csv:
            If save=True.

        .png:
            If plot=True, one ``plot_bypc`` figure per metric.
    """
    if group is None and folds is None:
        raise ValueError(
            "holdout_bypc requires at least one of group= (leave-one-group-out) or "
            "folds= (k-fold). Got both None."
        )
    if boot is not None and folds is None:
        raise ValueError(
            "boot= repeats the random partition (repeated K-fold CV), so it requires "
            "folds= (random or stratified K-fold). Leave-one-group-out (group= alone) "
            "has a deterministic split with nothing to resample."
        )

    drop_cols = [c for c in (group, cluster, groupby) if c is not None]
    feat_cols = df.columns.drop(drop_cols) if drop_cols else df.columns
    _check_rank(df[feat_cols])

    df_input = df.copy()
    if shuffle:
        from ...preprocessing.data_utils import fullmantel
        df_input[feat_cols] = fullmantel(df_input[feat_cols]).values

    cv, mode = _make_holdout_cv(group, folds, cluster, groupby=groupby)

    # Global anchor: a pooled-data PCA defines the canonical PC1..PC{npc} homologue.
    # Every per-fold PCA is Procrustes-aligned to this anchor (handled by rhom when
    # anchor= is set), so the bypc column index carries a consistent meaning across
    # folds. Same architecture as dir_proj_bypc. Under groupby the anchor and the
    # projection target are themselves within-group-standardized (grouped) solutions.
    anchor_input = (group_standardize(df_input, groupby, feature_cols=list(feat_cols))[list(feat_cols)]
                    if groupby else df_input[feat_cols])
    anchor_pca = basePCA(n_components=npc, rotation=rotation, method=method, corr=corr)
    anchor_pca.fit(anchor_input)
    anchor = anchor_pca.loadings.to_numpy()

    boot_model = RHom(rd=copy.deepcopy(anchor_input.values), bypc=True, n_comp=npc,
                      method=method, rotation=rotation, corr=corr, anchor=anchor)

    total_folds = folds if folds is not None else int(df_input[group].nunique())

    print(f"Running {'Bootstrapped ' if boot else ''}By-Component Held-Out CV "
          f"({mode}{f', B={boot}' if boot else ''})")

    rows = []
    # Per-component summary distributions: pooled fold scores without boot, per-repeat
    # per-component CV means with boot.
    comp_rhm = {k: [] for k in range(npc)}
    comp_phi = {k: [] for k in range(npc)}
    comp_sub = {k: [] for k in range(npc)}

    for label, rhm_block, phi_block, sub_block in _holdout_blocks(
            cv, df_input, boot_model, subspace, boot, total_folds, mode, progress, tag=" bypc"):
        # rhm_block / phi_block are lists of per-component score lists (one entry per
        # fold in the block); sub_block is a list of whole-solution floats (or None).
        for k in range(npc):
            rhm_k = [scores[k] for scores in rhm_block]
            phi_k = [scores[k] for scores in phi_block]

            if boot is None:
                # Per-(fold, comp) detail row (single fold -> zero-width CI). Subspace
                # is whole-solution, so the fold's value is broadcast across components.
                rows.append(_build_row(boot_model.n_comp, rhm_k, phi_k,
                                       sub_data=sub_block if subspace else None,
                                       metadata={"fold": label, "comp": k + 1}))
                comp_rhm[k] += rhm_k
                comp_phi[k] += phi_k
                if subspace:
                    comp_sub[k] += sub_block
            else:
                comp_rhm[k].append(float(np.mean(rhm_k)))
                comp_phi[k].append(float(np.mean(phi_k)))
                if subspace:
                    comp_sub[k].append(float(np.mean(sub_block)))

        if display:
            shown = ", ".join(
                f"PC{k+1}={np.mean([scores[k] for scores in rhm_block]):.3g}"
                for k in range(npc)
            )
            print(f"  {label}: rhm {shown}")

    # Per-component summary rows (across folds without boot, across repeats with boot).
    for k in range(npc):
        rows.append(_build_row(boot_model.n_comp, comp_rhm[k], comp_phi[k],
                               sub_data=comp_sub[k] if subspace else None,
                               metadata={"fold": "summary", "comp": k + 1}))

    holdout_bypc_df = pd.DataFrame(rows)
    # Attach anchor loadings so plot_bypc can render wordclouds without a disk round-trip.
    holdout_bypc_df.attrs["loadings"] = anchor_pca.loadings.copy()

    if plot:
        from ...visualization.rhomplots import plot_bypc
        setupanalysis(path, file_prefix, includetime=False)
        plt.close('all')

        metrics = ["rhm", "phi"] + (["sub"] if subspace else [])
        for name in metrics:
            fig = plot_bypc(holdout_bypc_df, group="fold", metric=name,
                            title=f"By-Component Held-Out CV ({mode}): {name.upper()}")
            fig.savefig(
                os.path.join(path, f"{file_prefix}/{file_prefix}_holdout_bypc_{len(feat_cols)}D_{npc}PC_{name}.png"),
                bbox_inches="tight", dpi=150,
            )
            plt.show()
            plt.close(fig)

        # Persist the anchor loadings alongside the stats CSV (same convention as the
        # other bypc builtins) so plot_bypc can be re-run from disk.
        loadings_path = os.path.join(
            path, f"{file_prefix}",
            f"{file_prefix}_loadings_{len(feat_cols)}D_{npc}PC.csv",
        )
        anchor_pca.loadings.to_csv(loadings_path)

    if save:
        _export_report(holdout_bypc_df, path, file_prefix,
                       f"holdout_bypc_{len(feat_cols)}D_{npc}PC")

    return holdout_bypc_df
