from .._deps import pd, np, Tuple, warnings, PCA, eigh, Rotator, Union

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

    # A full PCA already contains the top-n components, so slice them rather than
    # fitting a second restricted PCA (which doubled the SVD cost in every bootstrap
    # replicate). Identical to PCA(n_components=n).fit: that solver also computes the
    # full SVD and just truncates.
    # Loadings: eigenvectors * sqrt(eigenvalues)
    loadings = full_pca.components_[:n_components].T * np.sqrt(full_pca.explained_variance_[:n_components])

    return full_pca, loadings, eigenvalues

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
            Which correlation matrix to decompose; see :func:`correlation_matrix`.
    """
    from RHom.preprocessing.correlations import correlation_matrix
    R = correlation_matrix(df, corr=corr)

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
