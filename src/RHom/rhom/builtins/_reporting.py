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
