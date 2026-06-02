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
from .._deps import pd, np

from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize_scalar


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
