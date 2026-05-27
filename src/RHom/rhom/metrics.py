from .._deps import pd, np, BaseEstimator, KFold, BaseCrossValidator, pearsonr
from ..core.base_pca import basePCA

from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment

from ..preprocessing.data_utils import tcc


class rhom(BaseEstimator):
    """
    R-Homologue and Tucker's Congruence Coefficient
    -----------------------------------------------
    A class for calculating component similarity between two sets of data.

    Parameters
    ----------
        - rd: array-like or pd.DataFrame, default=None
            - The referent dataset for deriving components.
        - n_comp: int, default=None
            - The number of components to extract from the dataset.
        - method: str, default="svd"
            - Method for PCA implementation.
            - Can take:
            1. 'svd' (sklearn PCA(svd_solver = "full"), numpy, scipy, MATLAB, R (prcomp), Stata)
            2. 'eigen'(SPSS, R (psych, princomp), SAS, sklearn PCA(svd_solver="auto"))
        - rotation: str, default="Varimax"
            - The rotation method to use for the loadings. If None or False, no rotation is performed.
            - Supported methods are "varimax", "promax", "oblimin", "oblimax", "quartimin", "quartimax", and "equamax".
        - bypc: bool, default=False
            - Whether to save component similarity on a by-component basis.

    Attributes
    ----------
        - model_x: PCA
            - The PCA object for the referent dataset.
        - model_x2: PCA
            - The PCA object for the comparate dataset.
        - results: pd.DataFrame
            - The projected component scores for each set of components.

    Methods
    -------
        - get_params(deep=True):
            - Retrieve the parameters of the instance.

        - fit(x, y=None):
            - Fit the model to the provided referent and comparate data.

        - predict(y=None):
            - Predict the output based on the provided input data.

        - hom_pairs(cor_matrix):
            - Calculate the correlation of homologous pairs in the correlation matrix of two datasets \n\t\t(Mulholland et al., 2023; Everett, 1983).

        - pro_cong():
            - Perform procrustes congruence analysis.
   
    References
    ----------

    Mulholland, B., Goodall-Halliwell, I., Wallace, R., Chitiz, L., McKeown, B., Rastan, A., Poerio, G. L., \n\tLeech, R., Turnbull, A., Klein, A., Milham, M., Wammes, J. D., Jefferies, E., & Smallwood, J. (2023). \n\tPatterns of ongoing thought in the real world. Consciousness and Cognition, 114, 103530. https://doi.org/10.1016/j.concog.2023.103530 

    Everett, J. E. (1983). Factor Comparability As A Means Of Determining The Number Of Factors And Their Rotation. \n\tMultivariate Behavioral Research, 18(2), 197-218. https://doi.org/10.1207/s15327906mbr1802_5

    """
    def __init__(self, rd=None, n_comp=None, method='svd', rotation="varimax", bypc=False):
            self.rd = rd
            self.n_comp = n_comp
            self.method = method
            self.rotation = rotation
            self.bypc = bypc
            
            # Sub-models instantiated during fit
            self.model_x = None
            self.model_x2 = None
        
    def fit(self, X, y=None):
        """
        Fit the model to the provided data.

        This function fits the model to the provided input data 'X' and 'y' if applicable.

        Parameters
        ----------
            X : array-like or DataFrame, shape (n_samples, n_features)
                The input data for the referent dataset.
            y : array-like or DataFrame, shape (n_samples, n_targets), optional, default=None
                The input data for the comparate dataset.

        Returns
        -------
            None
        """

        use_rot = self.rotation if (self.n_comp and self.n_comp >= 2) else False

        self.model_x = basePCA(n_components=self.n_comp, verbosity=0, rotation=use_rot, method=self.method)
        self.model_x.fit(pd.DataFrame(X))

        self.model_x2 = basePCA(n_components=self.n_comp, verbosity=0, rotation=False, method=self.method)
        self.model_x2.fit(pd.DataFrame(y if y is not None else X))
    
        return self

    def predict(self, X=None):
        """
        Generate component scores based on projections from each dataset.

        Parameters
        ----------
            X : array-like, default=None
                The input data for prediction.

        Returns
        -------
            results : list
                A list containing the predicted outputs.
        """
        data = X if X is not None else self.rd

        if data is None:
            raise ValueError("No tracking data available. Pass a matrix to predict() or set 'rd' at initialization.")

        loadings_x = self.model_x.loadings.to_numpy()
        loadings_x2 = self.model_x2.loadings.to_numpy()

        if self.rotation:
            # Orthogonal Procrustes alignment
            R, _ = orthogonal_procrustes(loadings_x, loadings_x2)
            aligned_loadings_x2 = np.dot(loadings_x2, R.T)
            
            return [np.dot(data, loadings_x), np.dot(data, aligned_loadings_x2)]
        
        return [self.model_x.transform(data), self.model_x2.transform(data)]

    def hom_pairs(self,cor_matrix):
        """
        Calculate homologous components between the referent and comparate datasets.

        Parameters
        ----------
            cor_matrix : array-like, shape (n_features, n_features)
                The correlation matrix between the components for the referent and comparate datasets.

        Returns
        -------
            homologous_pairs : list or float
                If self.bypc is True, a list containing the correlation values of homologous pairs. If self.bypc is False, the mean correlation value of homologous pairs.
        """

        # Slice out the cross-block safely (handles both cross-corr and raw similarity square matrices)
        if cor_matrix.shape[0] == 2 * self.n_comp:
            matrix_block = np.abs(cor_matrix[-self.n_comp:, :self.n_comp])
        else:
            matrix_block = np.abs(cor_matrix)

        # Utilize Hungarian matching algorithm to maximize overall weight
        # (linear_sum_assignment minimizes, so we pass negative weights)
        row_ind, col_ind = linear_sum_assignment(-matrix_block)
        optimal_matches = matrix_block[row_ind, col_ind]

        if self.bypc:
            return list(optimal_matches)
        return float(np.mean(optimal_matches))

    def pro_cong(self):
        """
        Performs procrustes congruence analysis between the loadings matrices of the referent and comparate fitted PCA models. It calculates the Tucker's Congruence Coefficient (TCC) for all possible combinations of loadings pairs between the two models.

        Returns
        -------
            phi : list or float
                If self.bypc is True, a list containing the TCC values of homologous pairs. If self.bypc is False, the mean TCC value of homologous pairs.

        """
        loadings_X = self.model_x.loadings.to_numpy()
        loadings_x2 = self.model_x2.loadings.to_numpy()
        
        # Align frameworks using orthogonal Procrustes rotation
        R, _ = orthogonal_procrustes(loadings_X, loadings_x2)
        loadings_x2 = np.dot(loadings_x2, R.T)

        # Streamlined 2D matrix build using a clean list comprehension
        tcc_matrix = np.array([
            [tcc(loadings_X[:, i], loadings_x2[:, j]) for j in range(self.n_comp)]
            for i in range(self.n_comp)
        ])

        # Pass the constructed matrix to match optimal pairs
        return self.hom_pairs(tcc_matrix)

    def old_cv(self,data,cv=None):
        if not cv:
            cv = KFold()
        else:
            assert isinstance(cv,BaseCrossValidator)
        baseline = self.fit_transform(data)
        folds = []
        correlations = []
        for x,y in cv.split(data):
            self.fit(data.iloc[x])
            out = self.transform(data.iloc[y])
            folds.append(out)
            outv = self.check_inputs(out).values.ravel()
            bv = self.check_inputs(baseline.iloc[y]).values.ravel()
            correlations.append(pearsonr(outv,bv)[0])
        return correlations, folds

