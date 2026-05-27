#TODO update groupedPCA
from .._deps import pd, StandardScaler
from .. import basePCA

class groupedPCA(basePCA):
    """
    A class for performing grouped Principal Component Analysis (PCA).

    Args:
        grouping_col: The column to group by.
        n_components: The number of components to use.
        **kwargs: Additional keyword arguments.

    Attributes:
        grouping_col: The column to group by.

    Methods:
        z_score_byitem(df_dict) -> pd.DataFrame:
            Calculate the z-score of the dataframe.

        z_score_byitem_project(df_dict) -> pd.DataFrame:
            Calculate the z-score of the dataframe for projection.

        fit(df, y=None, **kwargs) -> self:
            Fit the grouped PCA model.

        transform(df) -> pd.DataFrame:
            Transform the input dataframe using the fitted grouped PCA model.

        save(savebygroup=False, path=None, pathprefix="analysis", includetime=True) -> None:
            Save the results of the grouped PCA analysis.

    """
    def __init__(self, grouping_col=None, n_components="infer", **kwargs):
        super().__init__(n_components)
        self.grouping_col = grouping_col
        if grouping_col is None:
            raise ValueError("Must specify a grouping column.")

    def z_score_byitem(self, df_dict) -> pd.DataFrame:
        """
        This function is used to calculate the z-score of the dataframe.

        Args:
            df_dict (dict): Dictionary of dataframes.

        Returns:
            pd.DataFrame: Dataframe with z-score.
        """
        self.scalerdict = {}
        outdict = []
        for key, value in df_dict.items():
            scaler = StandardScaler()
            value_ = self.check_inputs(value, fit=True)
            value_scaled = scaler.fit_transform(value_)
            extcol = self.extra_columns.copy().assign(
                **dict(zip(self.items, value_scaled.T))
            )
            self.scalerdict[key] = scaler
            outdict.append(extcol)
            
        return pd.concat(outdict, axis=0)

    def z_score_byitem_project(self, df_dict) -> pd.DataFrame:
        """
        This function takes a dictionary of dataframes and returns a dataframe with z-scored values.

        Args:
            df_dict (dict): A dictionary of dataframes.

        Returns:
            pd.DataFrame: A dataframe with z-scored values.
        """
        outdict = []
        for key, value in df_dict.items():
            value_ = self.check_inputs(value, project=True)
            try:
                scaler = self.scalerdict[key]
            except Exception:
                print(
                    f"Encountered a group in the data that wasn't seen while fitting: {key}. New group will be zscored individually."
                )
                scaler = StandardScaler()
                scaler.fit(value_)
            value_scaled = scaler.transform(value_)
            extcol = self.project_columns.copy().assign(
                **dict(zip(self.items, value_scaled.T))
            )
            outdict.append(extcol)
        return pd.concat(outdict, axis=0)

    def fit(self, df: pd.DataFrame, y=None, **kwargs):
        """
        Fit the grouped PCA model.

        Args:
            df (pd.DataFrame): The dataframe to fit.
            y (pd.Series): The target variable.
            **kwargs: Additional keyword arguments.

        Returns:
            self

        """
        self.ogdf = df.copy()
        d = dict(tuple(df.groupby(self.grouping_col)))
        zdf = self.z_score_byitem(d)
        super().fit(zdf, y=y, scale=False, **kwargs)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input dataframe using the fitted grouped PCA model.

        Args:
            df (pd.DataFrame): The input dataframe to be transformed.

        Returns:
            pd.DataFrame: The transformed dataframe.

        """
        d = dict(tuple(df.groupby(self.grouping_col)))
        zdf = self.z_score_byitem_project(d)
        return super().transform(zdf, scale=False)
    
    def save(self,savebygroup=False,path=None,pathprefix="analysis",includetime=True):
        """
    Save the results of the grouped PCA analysis.

    Args:
        savebygroup (bool, optional): Whether to save results by group. Defaults to False.
        path (str or None, optional): The path to save the results. If None, use the default path. Defaults to None.
        pathprefix (str, optional): The prefix for the path. Defaults to "analysis".
        includetime (bool, optional): Whether to include the timestamp in the path. Defaults to True.

    Returns:
        None

    Raises:
        NotImplementedError: If saving by group is not yet implemented.

    """
        self.path = setupanalysis(path,"grouped_"+pathprefix,includetime)
        if savebygroup:
            raise NotImplementedError("Saving by group is not yet implemented.")
            
        else:
            super().save()
