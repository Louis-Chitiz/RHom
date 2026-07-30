"""
Shared reporting helpers for the rhom builtins.

Every analysis in this package ultimately produces summary rows with mean / SE /
CI columns built from a bootstrap (or single-fold) distribution, and most of them
also need a progress bar around their inner loop and a uniform on-disk export.
The five helpers here are the package-wide source of truth for that pattern --
edit them once and every builtin's output schema updates consistently.
"""
from ..._deps import pd, np

import os

from ...io.save import setupanalysis


def _summary_stats(distribution, alpha=0.05):
    """Internal statistical helper to calculate confidence intervals."""
    distribution = np.asarray(distribution)
    mean = np.mean(distribution)
    std_err = np.std(distribution, ddof=1) if len(distribution) > 1 else 0.0

    # Calculate a simple normal distribution or percentile-based CI
    low_ci = np.percentile(distribution, (alpha / 2) * 100)
    high_ci = np.percentile(distribution, (1 - alpha / 2) * 100)

    return {
        "x": mean,
        "se": std_err,
        "LCI": low_ci,
        "UCI": high_ci,
    }


def _resolve_null_seed(seed):
    """
    Resolve the null's RNG seed to a concrete integer.

    An explicit ``seed`` is passed through unchanged; ``None`` draws a fresh 32-bit seed
    from OS entropy. Builtins call this before the null block and record the result in
    ``df.attrs["null_seed"]``, so even a default (unseeded) run reports the exact seed it
    used and can be reproduced later by passing it back as ``null_seed=``.
    """
    if seed is not None:
        return seed
    return int(np.random.SeedSequence().generate_state(1)[0])


def _standardize_within(A, labels):
    """Z-score each column within each level of ``labels`` (the groupby= behaviour)."""
    if labels is None:
        return A
    out = np.empty_like(A, dtype=float)
    labels = np.asarray(labels)
    for g in np.unique(labels):
        m = labels == g
        block = A[m]
        sd = block.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        out[m] = (block - block.mean(axis=0)) / sd
    return out


def _chance_reference(features, npc, method="svd", rotation="varimax", corr="pearson",
                      subspace=False, null_reps=200, progress=False, desc="Chance null",
                      *, anchor=None, bypc=False, groupby_labels=None,
                      group_labels=None, fold_indices=None, rd=None, seed=None):
    """
    Estimate the chance level of the similarity metrics by permutation.

    Each of the ``null_reps`` draws permutes every feature column independently (via
    ``column_mantel`` -- destroying cross-variable structure while preserving each item's
    marginal), applies the same within-group standardization the real analysis used,
    splits the rows, fits a PCA on each side, and scores their similarity with *the same
    estimator configuration* the caller used for the observed values. Returns the pooled
    per-metric chance level + 95% CI (plus per-component entries when ``anchor`` is set)
    via ``_null_summary``.

    The chance floor for rhm / phi / sub is essentially a property of ``npc`` and the
    number of items (the metrics pick a best match, so even random components score
    positively), not of which builtin produced the comparison -- so a single shared
    estimator gives every analysis a consistent, interpretable "above chance?" baseline.

    Parameters
    ----------
    features : pd.DataFrame or ndarray
        The raw feature columns. Shuffling happens before standardization, so pass the
        unstandardized frame even when ``groupby_labels`` is set.
    anchor : ndarray, optional
        The same ``anchor_loadings`` used by the bypc builtins. When set, the null is
        scored in anchor mode (diagonal readout, no Hungarian) and the returned dict
        carries per-component arrays under ``phi_bypc`` / ``rhm_bypc`` alongside the
        pooled scalars, so the estimator matches the observed statistic.
    bypc : bool
        Mirror the estimator's ``bypc`` flag. Implied by ``anchor`` but accepted
        separately for symmetry with ``RHom``.
    groupby_labels : array-like, optional
        Per-row levels of the builtin's ``groupby`` column, so the null is standardized
        exactly as the observed analysis was.
    group_labels : array-like, optional
        Per-row levels of the builtin's ``group`` column. When supplied with exactly two
        levels, the null splits on those levels rather than drawing random halves, so it
        inherits the real n imbalance between the two sources. Ignored otherwise, and
        overridden by ``fold_indices`` when both are given.
    fold_indices : iterable of (train_idx, test_idx), optional
        Positional index arrays describing the observed held-out folds (e.g. from
        ``pair_cv.holdout_index_split``). When given, rep ``i`` is scored on fold
        ``i % len(fold_indices)``, so the null averages over the *same* fold geometry the
        observed statistic did -- reproducing the leave-one-group-out asymmetry (a large
        train set vs a small held-out group) and k-fold train/test sizes rather than
        symmetric random halves. Takes precedence over ``group_labels``.
    rd : ndarray, optional
        Projection target for the R-homologue score. Defaults to the (standardized)
        shuffled matrix, which is deliberate: the null projects onto the *shuffled*
        loadings so that both sides of the comparison come from the same null world.
        The observed analyses instead pass a scaled real-data target via
        ``RHom(rd=rd_src)``; projecting real data onto null loadings would be an
        incoherent hybrid, so this asymmetry is intentional. Pass an explicit array only
        to mirror a non-default observed ``RHom(rd=...)``.
    seed : int, optional
        Seed for the permutation RNG, making the chance band reproducible across runs.
    """
    from ..metrics import RHom
    from ...preprocessing.data_utils import column_mantel

    rng = np.random.default_rng(seed)

    feats = features.values if hasattr(features, "values") else np.asarray(features)
    feats = np.asarray(feats, dtype=float)
    n = feats.shape[0]

    use_anchor = anchor is not None
    per_comp = bool(bypc or use_anchor)

    # Held-out fold geometry (train/test index pairs) takes precedence: rep i cycles
    # through the observed folds so the null inherits the same asymmetry (e.g. LOGO's
    # large-train vs small-test split). Materialize to a list; an empty one falls back.
    folds = list(fold_indices) if fold_indices is not None else None
    if not folds:
        folds = None

    # Otherwise, a fixed two-source split when the comparison was between two groups.
    split_mask = None
    if folds is None and group_labels is not None:
        lab = np.asarray(group_labels)
        levels = pd.unique(lab)
        if len(levels) == 2:
            split_mask = (lab == levels[0])

    rhm_vals, phi_vals, sub_vals = [], [], []
    for i in _progress_wrap(range(null_reps), total=null_reps, desc=desc, enabled=progress):
        shuffled = column_mantel(feats, rng=rng)
        shuffled = _standardize_within(shuffled, groupby_labels)

        if folds is not None:
            tr, te = folds[i % len(folds)]
            a, b = shuffled[tr], shuffled[te]
        elif split_mask is not None:
            a, b = shuffled[split_mask], shuffled[~split_mask]
        else:
            order = rng.permutation(n)
            half = max(1, n // 2)
            a, b = shuffled[order[:half]], shuffled[order[half:]]

        target = shuffled if rd is None else rd
        model = RHom(rd=target, n_comp=npc, method=method, rotation=rotation,
                     corr=corr, bypc=per_comp,
                     anchor=anchor if use_anchor else None)
        model.fit(a, b)
        preds = model.predict()
        corrs = np.corrcoef(preds[0], preds[1], rowvar=False)

        rhm_vals.append(model.hom_pairs(corrs))
        phi_vals.append(model.pro_cong())
        if subspace:
            s = model.subspace_sim()
            sub_vals.append(float(np.mean(s)) if isinstance(s, (list, tuple, np.ndarray)) else float(s))

    return _null_summary(rhm_vals, phi_vals, sub_vals if subspace else None,
                         per_comp=per_comp, npc=npc)


def _null_summary(rhm_data, phi_data=None, sub_data=None, per_comp=False, npc=None):
    """
    Summarize permutation-null metric values into a per-metric chance reference.

    Given the metric values produced by running an analysis on column-shuffled data
    (cross-variable structure destroyed), return ``{metric: {x, se, LCI, UCI}}`` -- the
    chance level and its 95% CI for each of rhm / phi / (sub). This is what the builtins
    attach to their results (``df.attrs["null"]``) and the plot helpers draw as a
    reference line / band, so "is my reproducibility above chance?" is answered in-figure
    rather than by a separate shuffled rerun.

    With ``per_comp=True`` the rhm / phi inputs are lists of per-component lists; the
    pooled scalar (mean across components, for backwards compatibility with plot helpers
    that expect one number) is reported alongside a ``*_bypc`` list of per-component
    dicts. Do not draw the pooled scalar as a per-component reference line unless the
    per-component values are actually flat -- check ``*_bypc`` first. Subspace is a
    whole-solution property and is never split per component.
    """
    def _pack(data):
        arr = np.asarray(data, dtype=float)
        if not per_comp:
            return _summary_stats(arr), None
        arr = arr.reshape(len(arr), -1)          # [reps x npc]
        pooled = _summary_stats(arr.mean(axis=1))
        bypc = [_summary_stats(arr[:, j]) for j in range(arr.shape[1])]
        return pooled, bypc

    out = {}
    for name, data in (("rhm", rhm_data), ("phi", phi_data), ("sub", sub_data)):
        if data is None:
            continue
        if name == "sub":
            out[name] = _summary_stats(np.asarray(data, dtype=float))
            continue
        pooled, bypc = _pack(data)
        out[name] = pooled
        if bypc is not None:
            out[f"{name}_bypc"] = bypc
    return out


def _build_row(n_comp, rhm_data, phi_data=None, sub_data=None, metadata=None):
    """Assembles a single unified row mapping for reporting metrics."""
    row = {"n_comp": f"{n_comp}PC" if isinstance(n_comp, int) else n_comp}
    if metadata:
        row.update(metadata)

    rhm_stats = _summary_stats(rhm_data)
    row.update({f"rhm_{k}": v for k, v in rhm_stats.items()})

    if phi_data is not None:
        phi_stats = _summary_stats(phi_data)
        row.update({f"phi_{k}": v for k, v in phi_stats.items()})

    if sub_data is not None:
        sub_stats = _summary_stats(sub_data)
        row.update({f"sub_{k}": v for k, v in sub_stats.items()})

    return row


def _display_stats(header, row):
    """
    Pretty-print one summary row -- the flat dict returned by ``_build_row`` (keys
    ``rhm_x`` / ``rhm_se`` / ``rhm_LCI`` / ``rhm_UCI``, and optionally ``phi_*`` /
    ``sub_*``). Every builtin's ``display=True`` branch passes such a row directly.
    """
    def _line(label, prefix):
        return (f"{label:<27}{row[f'{prefix}_x']:.3g} +/- {row[f'{prefix}_se']:.3g} "
                f"95% CI[{row[f'{prefix}_LCI']:.3g}, {row[f'{prefix}_UCI']:.3g}]")

    print(f"{header}:\n" + "*" * 20)
    print(_line("Mean Homologue Similarity:", "rhm"))
    if "phi_x" in row:
        print(_line("Mean Factor Congruence:", "phi"))
    if "sub_x" in row:
        print(_line("Mean Subspace Similarity:", "sub"))
    print("*" * 40)


def _export_report(df, path, prefix, suffix):
    """Handles directory confirmation and saves the summary CSV file safely."""
    setupanalysis(path, prefix, includetime=False)
    filename = f"{prefix}_{suffix}.csv"
    full_path = os.path.join(path, f"{prefix}", filename)
    df.to_csv(full_path, index=False)
    print(f"Dataframe saved to: {full_path}")


def _progress_wrap(iterable, total=None, desc=None, enabled=True):
    """
    Optional progress bar via ``tqdm.auto`` (notebook-aware).

    Wraps any iterable with a tqdm progress bar when ``enabled`` is True. Falls back
    silently to the unwrapped iterable when tqdm is not installed, so the package
    keeps tqdm as a soft dependency. Passing a known ``total`` lets tqdm report
    proportion-complete and ETA; without it the bar still works but shows only the
    running count.
    """
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)
