from RHom._deps import pd, np, Tuple, PCA, eigh, Rotator, Union

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
    # loadings = model.components_.T * np.sqrt(model.explained_variance_)
    loadings = model.components_.T * np.sqrt(model.explained_variance_)
    
    return model, loadings, eigenvalues

def run_eigen(df: pd.DataFrame, n_components: Union[str, int] = "infer", verbosity: int = 0):
    """
    Runs PCA via Eigendecomposition of the correlation matrix.
    """

    # Compute the correlation matrix
    R = np.corrcoef(df, rowvar=False)

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
