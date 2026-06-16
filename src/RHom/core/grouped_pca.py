from .._deps import pd, StandardScaler
from .. import basePCA
from ..preprocessing.validation import check_inputs
from ..io.save import setupanalysis

class groupedPCA(basePCA):
    """
    Grouped Principal Component Analysis
    ------------------------------------
    A variant of basePCA that standardizes each variable *within each group* before
    pooling the rows back together for a single decomposition. This removes between-group
    differences in each item's mean and scale, so the resulting components reflect
    within-group covariance structure rather than differences between groups.

    Args:
        grouping_col (str): The column to group by (required).
        n_components (int or "infer"): Number of components to keep. Defaults to "infer".
        verbosity (int): Verbosity level passed to basePCA. Defaults to 0.
        rotation (str or bool): Rotation method passed to basePCA. Defaults to "varimax".
        method (str): Decomposition method passed to basePCA ("svd" or "eigen"). Defaults to "svd".
        corr (str): Correlation type passed to basePCA for the eigen decomposition method:
            "pearson" (default), "spearman", or "polychoric". As in basePCA, non-Pearson
            values require method="eigen" (the svd path is Pearson-only). The within-group
            standardization happens first regardless; corr only selects which correlation
            matrix of the pooled, group-standardized data is decomposed.

    Attributes:
        grouping_col (str): The column used for grouping.
        scalerdict (dict): Per-group StandardScaler objects fitted during fit().

    Methods:
        fit(df, y=None, **kwargs) -> self:
            Group-standardize and fit the pooled PCA model.

        transform(df) -> pd.DataFrame:
            Group-standardize new data (reusing fitted scalers) and project it.

        save(savebygroup=False, path=None, pathprefix="analysis", includetime=True) -> None:
            Save the results of the grouped PCA analysis.
    """
    def __init__(self, grouping_col=None, n_components="infer", verbosity=0, rotation="varimax",
                 method="svd", corr="pearson"):
        if grouping_col is None:
            raise ValueError("Must specify a grouping column.")
        super().__init__(n_components=n_components, verbosity=verbosity, rotation=rotation,
                         method=method, corr=corr)
        self.grouping_col = grouping_col
        self.scalerdict = {}

    def _standardize_groups(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Z-score each numeric item within each group and return the pooled frame in the
        original row order. During fit, a StandardScaler is fitted and stored per group;
        during transform, the fitted scaler is reused (a group unseen at fit time is
        z-scored on its own, with a warning).
        """
        if fit:
            self.scalerdict = {}

        blocks = []
        for key, sub in df.groupby(self.grouping_col):
            features = sub.drop(columns=self.grouping_col)
            numeric_df, metadata_df, _ = check_inputs(features, fit=fit)

            if fit:
                scaler = StandardScaler().fit(numeric_df)
                self.scalerdict[key] = scaler
            else:
                scaler = self.scalerdict.get(key)
                if scaler is None:
                    print(f"Encountered a group not seen while fitting: {key}. "
                          "It will be z-scored on its own.")
                    scaler = StandardScaler().fit(numeric_df)

            scaled_df = pd.DataFrame(
                scaler.transform(numeric_df),
                index=numeric_df.index,
                columns=numeric_df.columns,
            )

            block = pd.concat([metadata_df, scaled_df], axis=1)
            # Re-attach the grouping label as metadata; object dtype keeps it out of the decomposition.
            block[self.grouping_col] = pd.Series(key, index=block.index, dtype=object)
            blocks.append(block)

        pooled = pd.concat(blocks, axis=0)
        return pooled.loc[df.index]

    def _raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The unstandardized numeric feature columns, with the grouping label dropped.

        Used only for the descriptive "average responses" figure. Group
        standardization forces each item's within-group mean to ~0, so plotting the
        standardized values gives empty bars and unit-length error bars; this falls
        back to the raw values so the figure shows real item response levels.
        """
        numeric_df, _, _ = check_inputs(df.drop(columns=self.grouping_col), fit=True)
        return numeric_df

    def fit(self, df: pd.DataFrame, y=None, **kwargs) -> "groupedPCA":
        """Group-standardize the data, then fit the pooled PCA via basePCA (scale=False)."""
        self.ogdf = df.copy()
        pooled = self._standardize_groups(df, fit=True)
        # scale=False: the data is already standardized within each group.
        super().fit(pooled, y=y, scale=False, **kwargs)
        # super().fit() stored the group-standardized values as _raw_fitted; replace
        # them with the raw features so the saved means plot is meaningful (see
        # _raw_features). Decomposition / scores are unaffected -- this attribute only
        # feeds the descriptive figure.
        self._raw_fitted = self._raw_features(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group-standardize new data with the fitted scalers, then project via basePCA."""
        pooled = self._standardize_groups(df, fit=False)
        projected = super().transform(pooled, scale=False)
        # Same reasoning as fit(): report raw item levels for the descriptive plot.
        self._raw_project = self._raw_features(df)
        return projected

    def save(self, savebygroup=False, path=None, pathprefix="analysis", includetime=True):
        """
        Save the results of the grouped PCA analysis.

        Args:
            savebygroup (bool): Whether to save results by group (not yet implemented).
            path (str or None): Output directory. If None, uses the default.
            pathprefix (str): Prefix for the output folder. Defaults to "analysis".
            includetime (bool): Whether to include a timestamp in the folder name.

        Raises:
            NotImplementedError: If saving by group is requested.
        """
        if savebygroup:
            raise NotImplementedError("Saving by group is not yet implemented.")

        self.path = setupanalysis(path, "grouped_" + pathprefix, includetime)
        super().save()
