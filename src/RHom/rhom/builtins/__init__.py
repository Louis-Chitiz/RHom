"""
RHom builtins -- user-facing reproducibility analyses.

This package collects the top-level analysis functions, grouped by family:

    splithalf      -- bootstrap-half reproducibility (aggregate + bypc)
    dir_proj       -- pairwise direct-projection reproducibility (aggregate + bypc)
    omni           -- omnibus-frame reproducibility (sample, bypc, variance attribution)
    holdout        -- held-out CV reproducibility (LOGO / k-fold / stratified)
    consensus      -- bootstrap-aggregated loadings with per-element CIs

The package's public surface (re-exported here) is unchanged from when this was a
single ``builtins.py``; existing imports like
``from RHom.rhom.builtins import splithalf`` continue to work.

Shared reporting helpers (``_build_row``, ``_summary_stats``, ``_display_stats``,
``_export_report``, ``_progress_wrap``) live in ``_reporting`` and are imported by
the analysis modules; they are not re-exported here (private to the package).
"""
from .splithalf import splithalf, splithalf_bypc
from .dir_proj import dir_proj, dir_proj_bypc
from .omni import omni_sample, omsamp_bypc, omni_variance
from .holdout import holdout_cv, holdout_bypc
from .consensus import consensus_pca

__all__ = [
    "splithalf",
    "splithalf_bypc",
    "dir_proj",
    "dir_proj_bypc",
    "omni_sample",
    "omsamp_bypc",
    "omni_variance",
    "holdout_cv",
    "holdout_bypc",
    "consensus_pca",
]
