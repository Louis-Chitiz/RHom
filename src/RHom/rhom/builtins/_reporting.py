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


def _chance_reference(features, npc, method="svd", rotation="varimax", corr="pearson",
                      subspace=False, null_reps=200, progress=False, desc="Chance null"):
    """
    Estimate the chance level of the similarity metrics by permutation.

    Each of the ``null_reps`` draws Mantel-shuffles the feature columns (destroying
    cross-variable structure while preserving each item's marginal), splits the rows
    into two random halves, fits a PCA on each, and scores their similarity exactly as
    the real analyses do. Returns the pooled per-metric chance level + 95% CI via
    ``_null_summary``.

    The chance floor for rhm / phi / sub is essentially a property of ``npc`` and the
    number of items (the metrics pick a best match, so even random components score
    positively), not of which builtin produced the comparison -- so a single shared
    estimator gives every analysis a consistent, interpretable "above chance?" baseline.
    """
    from ..metrics import RHom
    from ...preprocessing.data_utils import fullmantel

    feats = features.values if hasattr(features, "values") else np.asarray(features)
    feats = np.asarray(feats, dtype=float)
    n = feats.shape[0]
    half = max(1, n // 2)

    rhm_vals, phi_vals, sub_vals = [], [], []
    for _ in _progress_wrap(range(null_reps), total=null_reps, desc=desc, enabled=progress):
        shuffled = fullmantel(pd.DataFrame(feats)).values
        order = np.random.permutation(n)
        a, b = shuffled[order[:half]], shuffled[order[half:]]

        model = RHom(rd=shuffled, n_comp=npc, method=method, rotation=rotation, corr=corr)
        model.fit(a, b)
        preds = model.predict()
        corrs = np.corrcoef(preds[0], preds[1], rowvar=False)

        rhm_vals.append(float(model.hom_pairs(corrs)))
        phi_vals.append(float(model.pro_cong()))
        if subspace:
            s = model.subspace_sim()
            sub_vals.append(float(np.mean(s)) if isinstance(s, (list, tuple, np.ndarray)) else float(s))

    return _null_summary(rhm_vals, phi_vals, sub_vals if subspace else None)


def _null_summary(rhm_data, phi_data=None, sub_data=None):
    """
    Summarize pooled permutation-null metric values into a per-metric chance reference.

    Given the metric values produced by running an analysis on Mantel-shuffled data
    (cross-variable structure destroyed), return ``{metric: {x, se, LCI, UCI}}`` -- the
    chance level and its 95% CI for each of rhm / phi / (sub). This is what the builtins
    attach to their results (``df.attrs["null"]``) and the plot helpers draw as a
    reference line / band, so "is my reproducibility above chance?" is answered in-figure
    rather than by a separate shuffled rerun.
    """
    out = {"rhm": _summary_stats(rhm_data)}
    if phi_data is not None:
        out["phi"] = _summary_stats(phi_data)
    if sub_data is not None:
        out["sub"] = _summary_stats(sub_data)
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


def _display_stats(header, rhm_stats, phi_stats):
    print(f"{header}:\n" + "*" * 20)
    print(
        f"Mean Homologue Similarity: {rhm_stats['x']:.3g} +/- {rhm_stats['se']:.3g} "
        f"95% CI[{rhm_stats['LCI']:.3g}, {rhm_stats['UCI']:.3g}]"
    )
    if phi_stats:
        print(
            f"Mean Factor Congruence:    {phi_stats['x']:.3g} +/- {phi_stats['se']:.3g} "
            f"95% CI[{phi_stats['LCI']:.3g}, {phi_stats['UCI']:.3g}]"
        )
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
