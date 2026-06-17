from .._deps import pd, np, Tuple, PCA, eigh, Rotator, Union

import warnings

def run_svd(df: pd.DataFrame, n_components: Union[int, str] = "infer", verbosity: int = 0):
    """
    Runs PCA using SVD. Returns the model, loadings, and eigenvalues.
    """

    full_pca = PCA(svd_solver="full").fit(df)
    eigenvalues = full_pca.explained_variance_

    # Determine components if 'infer' is selected
    if n_components == "infer":
        n_components = np.sum(eigenvalues >= 1)
        # Fallback to at least 1 component if none meet the criterion
        n_components = max(1, int(n_components))
        if verbosity > 0:
            print(f"Inferred number of components: {n_components}")


    # Fit the actual model with restricted components
    model = PCA(n_components=n_components, svd_solver="full").fit(df)
    
    # Calculate loadings: eigenvectors * sqrt(eigenvalues)
    loadings = model.components_.T * np.sqrt(model.explained_variance_)
    
    return model, loadings, eigenvalues

def run_eigen(df: pd.DataFrame, n_components: Union[str, int] = "infer", verbosity: int = 0,
              corr: str = "pearson"):
    """
    Runs PCA via eigendecomposition of a correlation matrix.

    Parameters
    ----------
        df: pd.DataFrame or array-like
            The numeric data to decompose.
        n_components: int or "infer", default="infer"
            Number of components to retain. With "infer", uses the Kaiser >=1 rule.
        verbosity: int, default=0
        corr: str, default="pearson"
            Which correlation matrix to decompose:
              - "pearson" (default) -- numpy's product-moment correlation
              - "spearman" -- rank correlation; monotonic-invariant, ordinal-friendly
              - "polychoric" -- latent-continuous correlation behind ordinal items via
                Olsson (1979) MLE. Only meaningful for genuinely ordinal data; much
                slower (one optimisation per pair).
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

    # Perform eigen decomposition (eigh returns values in ascending order)
    evals, evecs = eigh(R)
    
    # Sort in descending order
    idx = np.argsort(evals)[::-1]
    eigenvalues = evals[idx]
    eigenvectors = evecs[:, idx]

    # Component Selection
    if n_components == "infer":
        n_components = np.sum(eigenvalues >= 1)
        n_components = max(1, int(n_components))
        if verbosity > 0:
            print(f"Inferred number of components: {n_components}")

    # Extract top eigenvectors and compute loadings
    selected_ev = eigenvectors[:, :n_components]
    loadings = selected_ev * np.sqrt(eigenvalues[:n_components])

    return eigenvalues, loadings


def rotation(loadings: np.ndarray, method: Union[str, bool] = "varimax"):
    """
    Applies the specified rotation to the loadings matrix.
    """
    supported_methods = [
        "varimax", "promax", "oblimin", "oblimax", 
        "quartimin", "quartimax", "equamax"
    ]
    
    if not method:
        return loadings

    if method.lower() in supported_methods:
        rotator = Rotator(method=method.lower(), normalize=True)
        return rotator.fit_transform(loadings)
    
    raise ValueError(f"Rotation '{method}' is not supported. Use: {supported_methods}")
