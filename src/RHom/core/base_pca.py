from .._deps import pd, randint, BaseEstimator, TransformerMixin, StandardScaler

from ..preprocessing.preliminary import check_stats
from ..preprocessing.data_utils import rename_special_columns, flip_loadings
from ..preprocessing.validation import check_inputs

from ..core.decomposition import run_svd, run_eigen, rotation
from ..io.save import run_save_sequence

class basePCA(TransformerMixin, BaseEstimator):
    """
    A base class for performing Principal Component Analysis (PCA) with RHom.

    Args:
        verbosity (int, optional): The level of verbosity. Set to 0 for no output, 1 for basic output, and 2 for detailed output. Defaults to 1.
        n_components (int or "infer", optional): The number of components to keep. If "infer", the number of components is determined based on the explained variance. Defaults to "infer".
        method (str, optional): The method to use for decomposition. Supported methods are "svd" (used in sklearn and R's prcomp()) and "eigen" (used in SPSS and R's princomp() and pca() from the psych library).
        rotation (str or bool, optional): The rotation method to use for the loadings. If False, no rotation is performed. Supported methods are "varimax", "promax", "oblimin", "oblimax", "quartimin", "quartimax", and "equamax". Defaults to "varimax".
        
    Attributes:
        n_components (int or "infer"): The number of components to keep.
        verbosity (int): The level of verbosity.
        rotation (str or bool): The rotation method to use for the loadings.
        path (str or None): The path to save the results.
        ogdf (pd.DataFrame or None): The original dataframe.
        scaler (StandardScaler): The scaler used for z-score normalization.
        loadings (pd.DataFrame): The loadings matrix.
        extra_columns (pd.DataFrame): The fitted PCA scores.
        project_columns (pd.DataFrame): The projected PCA scores.
        _raw_fitted (pd.DataFrame): The raw fitted data.
        _raw_project (pd.DataFrame): The raw projected data.
        fullpca (PCA): The PCA object with full components.
        items (list): The column names of the input dataframe.

    Methods:
        naive_pca(df: pd.DataFrame) -> Tuple[PCA, pd.DataFrame]:
            Perform PCA on the input dataframe.

        fit(df, y=None, scale: bool = True, **kwargs) -> "PCA":
            Fits the PCA model.

        transform(df: pd.DataFrame, scale=True) -> pd.DataFrame:
            Transform the input dataframe using the fitted PCA model.

        save(group=None, path=None, pathprefix="analysis", includetime=True) -> None:
            Save the results of the PCA analysis.

    """
    def __init__(self, n_components="infer", verbosity=0, rotation="varimax",
                 method='svd', corr='pearson'):
        self.n_components = n_components
        self.verbosity = verbosity
        self.rotation = rotation
        self.method = method
        self.corr = corr
        self.path = None
        self.ogdf = None

    def naive_pca(self, df):
        if self.method == "svd":
            if self.corr != "pearson":
                raise ValueError(
                    f"SVD path only supports corr='pearson'; got corr={self.corr!r}. "
                    "Use method='eigen' for 'spearman' or 'polychoric'."
                )
            self.fullpca, loadings, self.eigenvalues = run_svd(df, self.n_components, self.verbosity)
            
        elif self.method == "eigen":
            self.eigenvalues, loadings = run_eigen(df, self.n_components, self.verbosity,
                                                   corr=self.corr)
        
        if self.rotation:
            loadings = rotation(loadings, self.rotation)
        
        # Ensure n_components is updated if it was originally 'infer'
        self.n_components = loadings.shape[1]

        loadings = flip_loadings(loadings)
        
        return pd.DataFrame(
            loadings,
            index=self.items,
            columns=[f"PC{x+1}" for x in range(self.n_components)],
        )

    def fit(self, df: pd.DataFrame, y=None, scale: bool = True, **kwargs) -> "basePCA":
        """
        Fits the PCA model: Validates data, performs decomposition, and calculates scores.
        """
        # Standardize and Backup
        # _df = rename_special_columns(df.copy())
        _df = df.copy()
        if self.ogdf is None:
            self.ogdf = _df.copy()

        # Process Inputs & Extract Metadata
        numeric_df, metadata_df, self.items = check_inputs(_df, fit=True)
    
        # Store the metadata and numeric data in the class attributes
        self.extra_columns = metadata_df
        self._raw_fitted = numeric_df
        
        # Factorability & Scaling
        check_stats(numeric_df, self.verbosity)

        if scale:
            self.scaler = StandardScaler()
            # Returns a DataFrame with the same index/columns as numeric_df
            numeric_df = pd.DataFrame(self.scaler.fit_transform(numeric_df))

        # Decompose using naive_pca
        self.loadings = self.naive_pca(numeric_df)
        
        # Calculate & Store Scores (Vectorized)
        pca_scores = numeric_df.values @ self.loadings.values
        
        score_cols = [f"PCA_{i+1}" for i in range(self.n_components)]
        scores_df = pd.DataFrame(pca_scores, index=_df.index, columns=score_cols)
        
        # Merge PCA scores back with the metadata (non-numeric columns)
        self.extra_columns = pd.concat([self.extra_columns, scores_df], axis=1)

        # Reporting
        if self.verbosity > 0:
            from RHom.visualization.loadings import print_highestloadings
            from RHom.visualization.pcastats import display_explained_variance

            display_explained_variance(self)
            print_highestloadings(self.loadings)

        return self
    
    def transform(self, df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
        """
        Transforms new data using the fitted PCA model and returns scores merged with metadata.
        """
        # Standardize column names and preserve index
        _df = rename_special_columns(df.copy())
        new_index = _df.index if isinstance(_df, pd.DataFrame) else list(range(len(_df)))

        # Extract numeric data and metadata
        numeric_df, metadata_df, _ = check_inputs(_df, fit=False)
        
        self.project_columns = metadata_df
        self._raw_project = numeric_df

        # Apply the fitted scaler from fit() so new data is standardized against training stats
        if scale:
            numeric_df = pd.DataFrame(
                self.scaler.transform(numeric_df),
                index=new_index,
                columns=self.items,
            )

        # Project data onto components (Matrix Multiplication)
        # Projecting N samples onto K components: (N x P) @ (P x K)
        projected_matrix = numeric_df.values @ self.loadings.values
        
        # Create the scores DataFrame
        score_cols = [f"PCA_{i+1}" for i in range(self.n_components)]
        scores_df = pd.DataFrame(projected_matrix, index=new_index, columns=score_cols)

        # Merge with metadata (project_columns) and return
        self.project_columns = pd.concat([self.project_columns, scores_df], axis=1)
        
        return self.project_columns.copy()
                
    def save(self, path=None, pathprefix=randint(10000,99999)):
        run_save_sequence(self, path=path, pathprefix=pathprefix)
        
        if self.verbosity > 0:
            print(f"Results organized and saved to {self.path}")