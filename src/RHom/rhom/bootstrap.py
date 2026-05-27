from .._deps import pd, np, Any, Dict, Optional

from ..preprocessing.data_utils import fullmantel

class BootstrapEngine:
    """
    Handles execution of resampling loops for various reproducibility metrics.
    """
    def __init__(
            self,
            estimator: Any,
            cv: Any,
            omnibus: bool = False,
            splithalf: bool = False,
            pro_cong: bool = False,
            bypc: bool = False,
            shuffle: bool = False,
            fit_params: Optional[Dict[str, Any]] = None
        ) -> None:
            self.estimator = estimator
            self.cv = cv
            self.omnibus = omnibus
            self.splithalf = splithalf
            self.pro_cong = pro_cong
            self.bypc = bypc
            self.shuffle = shuffle
            self.fit_params = fit_params or {}

    def __call__(self, X: pd.DataFrame, y: Any, group: Optional[str] = None) -> Any:
        """Enables object instance to be called exactly like the original function."""
        
        # Handle Full-Mantel Shuffling if requested
        if self.shuffle:
            # # Fallback protects against omitted 'group' strings in orchestration calls
            # group_col = group or getattr(self.cv, 'group', None)
            X = pd.DataFrame(fullmantel(X))

        # Generate splits according to the specified CV strategy
        if self.omnibus or self.splithalf:
            splits = self.cv.redists(df=X, subset=y)
        elif self.bypc:
            splits = self.cv.bypc_split(X, y)
        else:
            splits = self.cv.split(X, y)

        # Obtains scores and optional phi values for each split
        scores, phis = [], []
        for x1, x2 in splits:
            self.estimator.fit(x1, x2, **self.fit_params)
            preds = self.estimator.predict()
            corrs = np.corrcoef(preds[0], preds[1], rowvar=False)
            
            scores.append(self.estimator.hom_pairs(corrs))
            if self.pro_cong:
                phis.append(self.estimator.pro_cong())

        # Output scores by component if requested
        if self.bypc:
            complist = list(map(list, zip(*scores)))
            if self.pro_cong:
                return [complist, list(map(list, zip(*phis)))]
            return complist

        # Return overall scores (and optional phis) as lists
        if self.pro_cong:
            return [scores, phis]  # Empty list preserves the original 'exp_var' index position
        return scores