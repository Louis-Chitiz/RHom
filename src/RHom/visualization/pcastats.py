from RHom._deps import pd, np, plt
from RHom.preprocessing.data_utils import unrotated_variance

def plot_stats(df:pd.DataFrame):

    means = df.mean()
    stds = df.std()
    question_names = means.index.tolist()

    fig, ax = plt.subplots(figsize=(8, 6))

    
    ax.bar(question_names,means,yerr=stds)
    ax.set_title("Average responses")
    ax.set_xlabel("Item")
    ax.set_ylabel("Average response")
    ax.tick_params(axis = 'x', labelrotation = 45)
    
    fig.tight_layout()
    
    return fig

def display_explained_variance(pca, verbosity:int=1) -> pd.DataFrame:
    """
    Returns a DataFrame containing variance statistics for both 
    unrotated (original) and rotated components.
    """

    report_df = unrotated_variance(pca)

    report_df = report_df.loc[:pca.n_components+1, :]

    # Rotated Statistics (Based on Sum of Squared Loadings)
    if pca.rotation:
        
        from RHom.preprocessing.data_utils import rotated_variance
        rotated_data = rotated_variance(pca)
        
        report_df = pd.merge(report_df, rotated_data, on = "Component")
    
    if verbosity > 0:
        print(report_df)

    return report_df

def plot_scree(pca, type: str = None):
    """
    Plot the scree plot of the PCA.

    :param pca: The PCA object.
    :param path: The path to save the plot.
    """

    unrotated_var = unrotated_variance(pca)

    type_dict = {'vals':unrotated_var['Eigenvalue'] if type == "eigenvals" else unrotated_var['Proportion'],
                 'ylabel':"Eigenvalues" if type == "eigenvals" else "Variance Explained"}

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        np.arange(1, len(unrotated_var) + 1),
        type_dict['vals'],
        "o-", linewidth=2, color="blue"
        )
    ax.set_title("Scree Plot")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel(type_dict['ylabel'])

    return fig

