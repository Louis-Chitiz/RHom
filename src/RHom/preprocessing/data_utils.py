from RHom._deps import pd, np, StandardScaler

def rename_special_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes domain-specific column naming."""
    if not isinstance(df, pd.DataFrame):
        return df
    rename_map = {
        col: col.lower().replace("focus", "Task").replace("other", "People")
        for col in df.columns if "focus" in col.lower() or "other" in col.lower()
    }
    return df.rename(columns=rename_map)

def flip_loadings(loadings: pd.DataFrame):
    # Ensure consistent sign (determinative PCA)
    # If the sum of loadings for a component is negative, flip it.
    signs = np.sign(np.sum(loadings, axis=0))
    loadings *= signs

    return loadings

def unrotated_variance(pca):

    if pca.eigenvalues is None:
        print("Model not fitted.")
        return pd.DataFrame()

    total_var = np.sum(pca.eigenvalues)

    # Unrotated Statistics (Based on raw eigenvalues)
    proportions = pca.eigenvalues / total_var
    cumulative = np.cumsum(proportions)

    # Create the Base Report
    data = pd.DataFrame({
        "Component":[f"PC{x+1}" for x in range(len(pca.eigenvalues))],
        "Eigenvalue": pca.eigenvalues,
        "Proportion": proportions,
        "Cumulative": cumulative
    })

    return data

def rotated_variance(pca):

    if pca.loadings is None:
        print("Model not fitted.")
        return pd.DataFrame()
    
    total_var = np.sum(pca.eigenvalues)

    # SSL = Sum of squared loadings per column
    ssl = np.sum(pca.loadings**2, axis=0)
    prop = ssl / total_var
    
    # Note: Cumulative variance for oblique rotations (Promax/Oblimin) 
    # is mathematically complex because components overlap. 
    # We report it here as the 'redistributed' cumulative sum.
    cumulative = np.cumsum(prop)

    data = pd.DataFrame({
        "Component":[f"PC{x+1}" for x in range(pca.n_components)],
        "Rotated_SSL": ssl,
        "Rotated_Proportion": prop,
        "Rotated_Cumulative": cumulative
    })

    return data

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

def fullmantel(df):
    """
    Full-Mantel Shuffle (Row and Column Permutation)
    -----------------------------------------------
    Detects numeric columns, shuffles their rows and columns independently,
    and returns a dataframe with non-numeric metadata preserved.
    """
    # Identify numeric vs non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    metadata_df = df.select_dtypes(exclude=[np.number])
    
    if numeric_df.empty:
        return df
        
    arr = numeric_df.values
    x, y = arr.shape
    
    # Perform the Mantel Shuffle
    # Shuffle rows first
    arr = arr[np.random.permutation(x)]
    
    # Shuffle columns independently for each row
    rows = np.indices((x, y))[0]
    cols = [np.random.permutation(y) for _ in range(x)]
    shuffled_values = arr[rows, cols]
    
    # Reconstruct the DataFrame
    shuffled_numeric = pd.DataFrame(
        shuffled_values, 
        index=numeric_df.index, 
        columns=numeric_df.columns
    )
    
    return pd.concat([metadata_df, shuffled_numeric], axis=1)