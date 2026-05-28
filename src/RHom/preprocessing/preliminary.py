from .._deps import pd, np, plt

import warnings
from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

def check_stats(df: pd.DataFrame, verbosity: int = 0) -> None:
    """
    This function checks the KMO and Bartlett Sphericity of the dataframe.
    Args:
        df: The dataframe to check.
    Returns:
        None
    """
    if verbosity == 0:
        return None
    
    bart = calculate_bartlett_sphericity(df)
    kmo = calculate_kmo(df)

    print(interpret_bartlett(bart, df))
    print(interpret_kmo(kmo))

def _check_rank(df_numeric):
    """
    Warns once if the full-sample data matrix is rank-deficient.

    A rank below the number of variables means the correlation matrix is singular,
    which destabilises the PCA decomposition (and is what triggers the per-fold
    Moore-Penrose warnings during resampling).
    """
    arr = df_numeric.values if hasattr(df_numeric, "values") else np.asarray(df_numeric)
    n_features = arr.shape[1]
    rank = np.linalg.matrix_rank(arr - arr.mean(axis=0))

    if rank < n_features:
        warnings.warn(
            f"Data matrix is rank-deficient (rank {rank} of {n_features} variables). "
            "Likely causes: collinear/duplicate items, a constant column, compositional/ipsative "
            "data summing to a constant, or fewer observations than variables. The PCA will still "
            "run, but trailing components carry no real variance and reproducibility scores may be "
            "unreliable.",
            stacklevel=2,
        )

def interpret_bartlett(bart, df: pd.DataFrame) -> None:
        
    interpretation_dict = {"status":"acceptable" if bart[1] < 0.05 else "unacceptable"}

    p = df.shape[1]
    degreesfreedom = p * (p - 1) / 2

    return f"χ2({int(degreesfreedom)}) = {bart[0]:.2f}, p = {bart[1]:.3f}.\nBartlett Sphericity is {interpretation_dict['status']}."

def interpret_kmo(kmo):
        
    k = kmo[1]
    
    interpretation_dict = {
        (0.9, 1): "marvelous",
        (0.8, 0.9): "meritorious",
        (0.7, 0.8): "middling",
        (0.6, 0.7): "mediocre",
        (0.5, 0.6): "miserable",
        (0, 0.5): "unacceptable"
    }

    interpretation = next((msg for (min_s, max_s), msg in interpretation_dict.items() if min_s <= k <= max_s), "KMO is invalid")

    return f"KMO = {k:.2f}, which is {interpretation} for dimension reduction."

def parallel_analysis(df, n_iter: int = 1000, percentile: float = 95, plot: bool = True, seed: int = None):
    """
    Horn's Parallel Analysis
    ------------------------
    Estimates the number of components to retain by comparing the eigenvalues of the
    observed correlation matrix against the eigenvalue distribution of random data of
    the same shape. A component is retained when its observed eigenvalue exceeds the
    chosen percentile of the random eigenvalues for that position.

    Parameters
    ----------
        df: pd.DataFrame or array-like
            The dataset to analyse (rows = observations, columns = variables). Non-numeric
            columns are dropped automatically.
        n_iter: int, default=1000
            Number of random datasets to generate for the null distribution.
        percentile: float, default=95
            Percentile of the random eigenvalue distribution used as the retention
            threshold (Horn's original method uses the mean; 95 is the modern recommendation).
        plot: bool, default=True
            If True, overlay observed vs random eigenvalues on a scree-style plot.
        seed: int, default=None
            Seed for reproducibility of the random draws.

    Returns
    -------
        dict
            Keys: 'n_components' (suggested retention), 'observed', 'random_mean',
            'random_pct' (eigenvalue arrays, descending), and 'figure'
            (matplotlib Figure, or None when plot=False).

    References
    ----------
        Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis.
            Psychometrika, 30(2), 179-185. https://doi.org/10.1007/BF02289447
    """
    if isinstance(df, pd.DataFrame):
        data = df.select_dtypes(include=[np.number]).to_numpy()
    else:
        data = np.asarray(df, dtype=float)

    n, p = data.shape

    # Observed eigenvalues of the correlation matrix, sorted descending
    observed = np.sort(np.linalg.eigvalsh(np.corrcoef(data, rowvar=False)))[::-1]

    # Null distribution: eigenvalues from random normal data of identical shape
    rng = np.random.default_rng(seed)
    rand_evals = np.empty((n_iter, p))
    for i in range(n_iter):
        rand = rng.standard_normal((n, p))
        rand_evals[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(rand, rowvar=False)))[::-1]

    random_mean = rand_evals.mean(axis=0)
    random_pct = np.percentile(rand_evals, percentile, axis=0)

    # Retain components whose observed eigenvalue clears the random threshold
    n_components = int(np.sum(observed > random_pct))

    fig = None
    if plot:
        comps = np.arange(1, p + 1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(comps, observed, "o-", color="#4C72B0", label="Observed data")
        ax.plot(comps, random_pct, "s--", color="#C44E52", label=f"Random ({percentile:g}th pct)")
        ax.plot(comps, random_mean, ":", color="0.5", label="Random (mean)")
        ax.axvline(n_components + 0.5, color="0.3", linewidth=1)
        ax.set_title(f"Parallel Analysis: retain {n_components} component{'s' if n_components != 1 else ''}")
        ax.set_xlabel("Component")
        ax.set_ylabel("Eigenvalue")
        ax.set_xticks(comps)
        ax.legend()
        fig.tight_layout()

    return {
        "n_components": n_components,
        "observed": observed,
        "random_mean": random_mean,
        "random_pct": random_pct,
        "figure": fig,
    }