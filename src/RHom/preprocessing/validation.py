from .._deps import pd, np, Tuple, List, Union, Any, npt


def check_inputs(
    df: Union[pd.DataFrame, npt.NDArray[Any]],
    fit: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Validates input data, separates numeric features from metadata, 
    and extracts feature names.

    Args:
        df: The input data (DataFrame or NumPy array).
        fit: Whether the function is being called during the fit stage.

    Returns:
        Tuple containing:
            - numeric_df: DataFrame containing only numeric data for PCA.
            - metadata_df: DataFrame containing non-numeric columns (IDs, labels).
            - item_names: List of strings representing the feature/column names.
    """

    # Ensure input is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        # Generate generic names if input is a raw array
        item_names = [f"item_{x}" for x in range(df.shape[1])]
        df = pd.DataFrame(df, columns=item_names)
    else:
        item_names = df.columns.tolist()

    # Separate numeric and non-numeric data using vectorized selection
    # This replaces manual loops and handles all int/float variants
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    metadata_df = df.select_dtypes(exclude=[np.number]).copy()

    # Update item_names to reflect only the numeric features used for PCA
    numeric_item_names = numeric_df.columns.tolist()

    return numeric_df, metadata_df, numeric_item_names