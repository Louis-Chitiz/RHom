"""
Correlation-matrix helpers for the `corr=...` option on `run_eigen` / `basePCA`.

Pearson and Spearman come straight from numpy/pandas. Polychoric is implemented here
because the field doesn't have a maintained pure-Python implementation: it is the
two-step MLE from Olsson (1979), with thresholds fixed from the marginal cumulative
proportions and only the latent correlation optimised.

Polychoric is much slower than Pearson/Spearman -- one bounded scalar optimisation
per pair, with several bivariate-normal CDF evaluations per optimisation step. It is
meaningful only on genuinely ordinal items (Likert): applied to continuous data it
will treat every unique value as a category, which is rarely what you want.
"""
from .._deps import pd, np, warnings

from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize_scalar

def tcc(fac1=None, fac2=None):
    """
    Tucker's Congruence Coefficient
    -------------------------------
    Calculate the Tucker's Congruence Coefficient (TCC) between two components, which is an estimate of their loading similarity.

    Parameters
    ----------
        fac1 : array-like
            The first component.
        fac2 : array-like
            The second component.

    Returns
    -------
        tcc : float
            The Tucker's Congruence Coefficient (TCC) between the two factors.

    Notes
    -----
        - The TCC is a measure of loading similarity between two given components (Tucker, 1951).
        - Lovik et al. (2020) suggest using the absolute value of the numerator for factor matching.

    References
    ----------
        Tucker, L. R. (1951). A method for synthesis of factor analysis studies (PRS-984). Washington, DC: Department of the Army. 

        Lovik, A., Nassiri, V., Verbeke, G., & Molenberghs, G. (2020). A modified tucker’s congruence coefficient for factor matching.
            Methodology: European Journal of Research Methods for the Behavioral and Social Sciences,
            16(1), 59-74. https://doi.org/10.5964/meth.2813 

    """
    numerator = np.abs(np.dot(fac1, fac2))
    denominator = np.linalg.norm(fac1) * np.linalg.norm(fac2)
    return numerator / denominator


def _bvn_lower_tail(h, k, rho):
    """
    P(X <= h, Y <= k) for the standard bivariate normal with correlation rho.

    Infinities must be handled explicitly -- scipy's multivariate_normal.cdf can
    return NaN for +/-inf bounds, which silently breaks the cell-probability
    inclusion-exclusion below and collapses the polychoric MLE to a constant.
    """
    if np.isneginf(h) or np.isneginf(k):
        return 0.0
    if np.isposinf(h) and np.isposinf(k):
        return 1.0
    if np.isposinf(h):
        return float(norm.cdf(k))
    if np.isposinf(k):
        return float(norm.cdf(h))
    if rho == 0:
        return float(norm.cdf(h) * norm.cdf(k))
    return float(multivariate_normal.cdf([h, k], mean=[0, 0], cov=[[1, rho], [rho, 1]]))


def polychoric_corr(x, y):
    """
    Maximum-likelihood polychoric correlation between two ordinal series (Olsson 1979).

    Thresholds are fixed from the marginal cumulative proportions (the standard
    two-step estimator); only rho is optimised. The bivariate-normal CDF is evaluated
    once at every corner of the contingency table for each candidate rho, then reused
    via inclusion-exclusion across cells -- ~4x cheaper than computing per-cell.
    """
    x = np.asarray(x); y = np.asarray(y)
    xu, yu = np.sort(np.unique(x)), np.sort(np.unique(y))
    nx, ny = len(xu), len(yu)
    n = len(x)

    xi = np.searchsorted(xu, x)
    yi = np.searchsorted(yu, y)
    table = np.zeros((nx, ny), dtype=int)
    np.add.at(table, (xi, yi), 1)

    cum_x = np.cumsum(table.sum(axis=1) / n)[:-1]
    cum_y = np.cumsum(table.sum(axis=0) / n)[:-1]
    tau_x = np.concatenate([[-np.inf], norm.ppf(cum_x), [np.inf]])
    tau_y = np.concatenate([[-np.inf], norm.ppf(cum_y), [np.inf]])

    nonzero = np.argwhere(table != 0)

    def neg_log_lik(rho):
        # Precompute F at every corner -- (nx+1) x (ny+1) values, reused below.
        F = np.empty((nx + 1, ny + 1))
        for i in range(nx + 1):
            for j in range(ny + 1):
                F[i, j] = _bvn_lower_tail(tau_x[i], tau_y[j], rho)

        ll = 0.0
        for i, j in nonzero:
            p = F[i + 1, j + 1] - F[i, j + 1] - F[i + 1, j] + F[i, j]
            if p < 1e-300:
                p = 1e-300
            ll += table[i, j] * np.log(p)
        return -ll

    res = minimize_scalar(neg_log_lik, bounds=(-0.99, 0.99),
                          method="bounded", options={"xatol": 1e-4})
    return float(res.x)


def polychoric_corr_matrix(df, verbose=False):
    """
    Pairwise polychoric correlation matrix for an ordinal DataFrame or array.

    Each variable's distinct values are treated as ordered categories. With p items
    you pay p*(p-1)/2 polychoric optimisations -- expect seconds-to-minutes on a
    14-item ESQ matrix, hours on anything you'd plug into a bootstrap.

    Returns a DataFrame if `df` is a DataFrame, an ndarray otherwise.
    """
    is_df = hasattr(df, "values")
    arr = df.values if is_df else np.asarray(df)
    cols = df.columns.tolist() if is_df else list(range(arr.shape[1]))
    p = arr.shape[1]

    R = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            R[i, j] = R[j, i] = polychoric_corr(arr[:, i], arr[:, j])
            if verbose:
                print(f"  {cols[i]} x {cols[j]}: {R[i, j]:+.3f}")

    return pd.DataFrame(R, index=cols, columns=cols) if is_df else R

def correlation_matrix(df: pd.DataFrame, corr: str = "pearson") -> np.ndarray:
    """
    Builds the correlation matrix decomposed by the eigen-based routines.

    Parameters
    ----------
        df: pd.DataFrame or array-like
            The numeric data.
        corr: str, default="pearson"
            Which correlation matrix to build:
              - "pearson" (default) -- numpy's product-moment correlation
              - "spearman" -- rank correlation; monotonic-invariant, ordinal-friendly
              - "polychoric" -- latent-continuous correlation behind ordinal items via
                Olsson (1979) MLE. Only meaningful for genuinely ordinal data; much
                slower (one optimisation per pair).

    Returns
    -------
        np.ndarray
            The (p x p) correlation matrix. Non-finite entries from zero-variance
            columns (e.g. a constant item within a small resampling fold) are zeroed
            with a warning, so a downstream eigendecomposition can proceed instead of
            aborting.
    """
    if not isinstance(corr, str):
        raise TypeError(
            f"corr must be a string ('pearson', 'spearman', or 'polychoric'); "
            f"got {type(corr).__name__}."
        )
    corr = corr.lower()
    if corr == "pearson":
        R = np.corrcoef(df, rowvar=False)
    elif corr == "spearman":
        R = pd.DataFrame(df).corr(method="spearman").values
    elif corr == "polychoric":
        from ..preprocessing.correlations import polychoric_corr_matrix
        R = polychoric_corr_matrix(df)
        R = R.values if hasattr(R, "values") else R
    else:
        raise ValueError(
            f"Unknown corr={corr!r}; pick one of 'pearson', 'spearman', 'polychoric'."
        )

    # Guard against non-finite correlations: a column with zero variance within this
    # subset (e.g. a constant item in a small resampling fold, common under spearman)
    # makes its correlations undefined (NaN), which eigh rejects outright. Zero out the
    # non-finite entries: a fully-constant item then becomes a zero row/column (a
    # zero-variance dimension -> eigenvalue 0 -> excluded from the top components, so it
    # carries ~zero loading), while an isolated bad pair is just treated as uncorrelated.
    # This lets the decomposition proceed instead of aborting the whole resampling run.
    if not np.all(np.isfinite(R)):
        n_bad = int(np.sum(~np.isfinite(np.diag(R))))
        warnings.warn(
            f"Correlation matrix had non-finite entries from {n_bad} zero-variance "
            "column(s) (e.g. a constant item within a resampling fold). They were "
            "zeroed so the eigendecomposition can proceed; the affected items carry "
            "~zero loading and drop out of the retained components. Frequent warnings "
            "suggest folds too small for the data (try fewer folds, or method='svd').",
            stacklevel=2,
        )
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    return R
