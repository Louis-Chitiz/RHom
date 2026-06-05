"""
Held-out cross-validation reproducibility.

Single builtin (``holdout_cv``) with a three-way mode toggle: LOGO, random K-fold,
or stratified K-fold by a grouping variable. Reuses ``rhom`` unchanged on train /
test pairs from ``pair_cv.holdout_split``.
"""
from ..._deps import pd, np, randint, plt

import os
import copy

from ...preprocessing.preliminary import _check_rank
from ...io.save import setupanalysis

from ..metrics import RHom
from ..resampling import pair_cv

from ._reporting import _build_row, _display_stats, _export_report, _progress_wrap


def holdout_cv(df=None, group=None, folds=None, npc=None, method='svd', rotation='varimax',
               corr='pearson', cluster=None, save=True, plot=True, display=False,
               shuffle=False, subspace=False, progress=True, path='results',
               file_prefix=randint(10000, 99999)):
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

    Each fold returns point estimates (no bootstrap inside a fold); the final
    "summary" row holds the mean and across-fold CI of each metric.

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

        cluster: str, default=None
            Optional level-2 / clustering column. With ``folds=K`` (random or
            stratified), whole clusters are kept together within a fold (requires at
            least K distinct clusters; in stratified mode, K within each stratum).
            With ``group=...`` only, cluster is irrelevant since groups already define
            the partition.

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
            One row per fold with ``rhm_x`` / ``phi_x`` (and ``sub_x`` if requested)
            point estimates, plus a final ``fold='summary'`` row holding the mean
            and across-fold CI for each metric.

        .csv:
            If save=True.

        .png:
            If plot=True, one horizontal-bar figure per metric via ``plot_omni``.
    """
    if group is None and folds is None:
        raise ValueError(
            "holdout_cv requires at least one of group= (leave-one-group-out) or "
            "folds= (k-fold). Got both None."
        )

    drop_cols = [c for c in (group, cluster) if c is not None]
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

    # Mode dispatch: pair_cv's stratified_kfold flag flips holdout_split into the
    # stratified-K-fold path; otherwise self.group toggles LOGO and falling through
    # gives plain K-fold.
    if group is not None and folds is not None:
        cv = pair_cv(k=folds, cluster=cluster, stratify=group, stratified_kfold=True)
        mode = f"{folds}-fold stratified by '{group}'"
    elif group is not None:
        cv = pair_cv(group=group, cluster=cluster)
        mode = f"leave-one-group-out by '{group}'"
    else:
        cv = pair_cv(k=folds, cluster=cluster)
        mode = f"{folds}-fold"

    boot_model = RHom(rd=copy.deepcopy(df_input[feat_cols].values), n_comp=npc,
                      method=method, rotation=rotation, corr=corr)

    rows = []
    rhm_vals, phi_vals, sub_vals = [], [], []

    # Total folds: K for k-fold modes (random or stratified), G for LOGO.
    total_folds = folds if folds is not None else int(df_input[group].nunique())

    print(f"Running Held-Out CV ({mode})")
    fold_iter = _progress_wrap(
        cv.holdout_split(df_input),
        total=total_folds,
        desc=f"Held-out CV ({mode})",
        enabled=progress,
    )
    for train, test, label in fold_iter:
        boot_model.fit(train, test)
        preds = boot_model.predict()
        corrs = np.corrcoef(preds[0], preds[1], rowvar=False)

        rhm_val = float(boot_model.hom_pairs(corrs))
        phi_val = float(boot_model.pro_cong())
        sub_val = float(boot_model.subspace_sim()) if subspace else None

        rhm_vals.append(rhm_val)
        phi_vals.append(phi_val)
        if subspace:
            sub_vals.append(sub_val)

        # Per-fold row: pass a single-value "distribution" to _build_row so the schema
        # matches the summary row exactly (CIs and SE collapse to the point estimate;
        # zero-width error bars on the plot mark these as deterministic per-fold scores).
        meta = {"fold": label}
        row = _build_row(boot_model.n_comp, [rhm_val], [phi_val],
                         sub_data=[sub_val] if subspace else None, metadata=meta)
        rows.append(row)

        if display:
            extras = f", sub={sub_val:.3g}" if subspace else ""
            print(f"  held out '{label}': rhm={rhm_val:.3g}, phi={phi_val:.3g}{extras}")

    # Cross-fold summary: same _build_row call but fed the actual fold-level distribution,
    # so rhm_se / rhm_LCI / rhm_UCI report the across-fold spread.
    summary_row = _build_row(
        boot_model.n_comp, rhm_vals, phi_vals,
        sub_data=sub_vals if subspace else None,
        metadata={"fold": "summary"},
    )
    rows.append(summary_row)

    if display:
        _display_stats(f"Held-Out CV summary ({mode})", summary_row)

    holdout_df = pd.DataFrame(rows)

    if plot:
        from ...visualization.rhomplots import plot_omni
        setupanalysis(path, file_prefix, includetime=False)
        plt.close('all')

        metrics = ["rhm", "phi"] + (["sub"] if subspace else [])
        for name in metrics:
            # plot_omni's "Total" guard drops the omni_sample summary row; here we want
            # the summary row visible (it's the headline number), so we pass group='fold'
            # explicitly and rely on the label being 'summary' (not 'Total').
            fig = plot_omni(holdout_df, group="fold", metric=name,
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
