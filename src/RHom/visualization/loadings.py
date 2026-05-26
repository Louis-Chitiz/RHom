from RHom._deps import pd

def returnhighest(series: pd.Series, n: int) -> str:
    """
    Returns a formatted string of the top n features by absolute loading,
    grouped by their original sign.
    """
    # Get the top n features based on absolute magnitude
    top_indices = series.abs().nlargest(n).index
    top_loadings = series.loc[top_indices]

    # Separate into positive and negative sets
    pos_items = top_loadings[top_loadings > 0].index.tolist()
    neg_items = top_loadings[top_loadings < 0].index.tolist()

    pos_str = f"Positive_{'_'.join(pos_items)}" if pos_items else ""
    neg_str = f"Negative_{'_'.join(neg_items)}" if neg_items else ""

    # Handle cases where one group might be empty
    if not neg_str:
        return pos_str
    if not pos_str:
        return neg_str

    # Order the output based on which group has the higher average magnitude
    if top_loadings[top_loadings > 0].mean() > abs(top_loadings[top_loadings < 0].mean()):
        return f"{pos_str}_{neg_str}"
    else:
        return f"{neg_str}_{pos_str}"


def print_highestloadings(loadings: pd.DataFrame) -> None:
    
    if loadings is not None:

        for col in loadings:
            highest = returnhighest(loadings[col], 3)
            print (f"{col}: {highest}")

    else:
        print("PCA model not fitted. Please fit the model first.")

        