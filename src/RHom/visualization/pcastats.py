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

def _icc_a1(ratings: np.ndarray) -> float:
    """
    Two-way random-effects, absolute-agreement, single-measures ICC -- ICC(2,1) / ICC(A,1).

    `ratings` has shape (n_targets, n_raters); here that is (n_components, n_groups). Returns
    NaN when there are fewer than two components or two groups (ICC is undefined).
    """
    n, k = ratings.shape
    if n < 2 or k < 2:
        return np.nan

    grand = ratings.mean()
    row_means = ratings.mean(axis=1)
    col_means = ratings.mean(axis=0)

    ss_total = ((ratings - grand) ** 2).sum()
    ss_row = k * ((row_means - grand) ** 2).sum()       # between components
    ss_col = n * ((col_means - grand) ** 2).sum()        # between groups
    ss_err = ss_total - ss_row - ss_col

    msr = ss_row / (n - 1)
    msc = ss_col / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))

    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    return np.nan if denom == 0 else (msr - mse) / denom

def plot_grouped_scree(df: pd.DataFrame, group: str, type: str = None):
    """
    Grouped scree plot: overlays one scree slope per level of a grouping variable.

    For each group, the eigenvalues of the correlation matrix are computed from that
    group's data and plotted against component number, so the slopes can be compared
    directly across groups. The plot is annotated with an absolute-agreement ICC(2,1)
    summarising how closely the groups agree on the eigenvalue profile (1.0 = identical
    profiles; penalises both shape and magnitude disagreement, unlike plain correlation,
    which sits near 1 for any monotonic scree curve).

    :param df: DataFrame containing the decomposition columns and the grouping column.
    :param group: Name of the grouping column.
    :param type: "eigenvals" to plot raw eigenvalues (adds a Kaiser line at 1);
                 anything else plots proportion of variance explained.
    :return: matplotlib Figure.
    """
    ylabel = "Eigenvalues" if type == "eigenvals" else "Variance Explained"

    fig, ax = plt.subplots(figsize=(8, 6))

    slopes = []
    n_comp = 0
    for g in df[group].unique():
        numeric = df[df[group] == g].drop(columns=group).select_dtypes(include=[np.number])

        # Correlation-matrix eigenvalues (descending) make groups comparable regardless of n
        evals = np.sort(np.linalg.eigvalsh(np.corrcoef(numeric.to_numpy(), rowvar=False)))[::-1]
        vals = evals if type == "eigenvals" else evals / evals.sum()
        n_comp = len(vals)
        slopes.append(vals)

        ax.plot(np.arange(1, n_comp + 1), vals, "o-", linewidth=2, label=str(g))

    if type == "eigenvals":
        ax.axhline(1, color="0.5", linestyle="--", linewidth=1)  # Kaiser criterion

    # Annotate scree-slope agreement across groups: components = targets, groups = raters
    icc = _icc_a1(np.column_stack(slopes))
    icc_txt = "n/a" if np.isnan(icc) else f"{icc:.2f}".replace("-0.", "-.").lstrip("0") or "0"
    ax.text(0.125, 0.05, f"ICC = {icc_txt}", transform=ax.transAxes, ha="right", va="top",
            fontsize=11, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_title("Grouped Scree Plot")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(1, n_comp + 1))
    ax.legend(title=group)
    fig.tight_layout()

    return fig

