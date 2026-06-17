from .._deps import pd, np, Any, Dict, Optional

import warnings

from ..preprocessing.data_utils import fullmantel

class BootstrapEngine:
    """
    Handles execution of resampling loops for various reproducibility metrics.

    The engine has two orthogonal axes:

    1. ``mode`` controls the *resampling profile* (which ``pair_cv`` method generates
       the (x1, x2) pairs that are iterated):

           * ``None`` (default) -- ``cv.split(X, y)`` -- symmetric two-sided fold cross
             (used by ``dir_proj`` and ``dir_proj_bypc``).
           * ``"omnibus"`` or ``"splithalf"`` -- ``cv.resample_pairs(df=X, subset=y)`` --
             bootstrap half-resampling (used by ``omni_sample`` and ``splithalf``).
           * ``"asym"`` -- ``cv.asym_split(X, y)`` -- asymmetric X-fixed / y-folded
             split (used by ``omsamp_bypc``).

    2. ``per_component`` controls the *output shape*: when True, scores / phis come
       back as per-component lists (``[npc × n_replicates]``) and the subspace
       similarity is collapsed to one mean per replicate. When False, scores / phis
       are aggregate vectors of length ``n_replicates``. This flag is independent
       of the resampling profile -- any ``mode`` can be combined with either output
       shape, which is why ``dir_proj_bypc`` and ``splithalf_bypc`` can use
       ``mode=None``/``"splithalf"`` with ``per_component=True`` to get per-component
       results without the asymmetric resampling that ``omsamp_bypc`` needs.

    Parameters
    ----------
    estimator : Any
        An instance of a fitted RHom estimator that implements reproducibility computations.
    cv : Any
        An instance of a CV splitter that yields resamples/folds according to the desired strategy.
    mode : Optional[str], default None
        Resampling profile selector. See class-level docstring for the four values.
    per_component : bool, default False
        If True, transpose the score / phi accumulators into per-component lists and
        collapse the subspace cosines to per-replicate means.
    shuffle : bool, default False
        If True, applies Full-Mantel Shuffling to the input data before splitting.
    pro_cong : bool, default False
        If True, computes Tucker's phi values for each split and returns them alongside scores.
    subspace : bool, default False
        If True, computes subspace similarity for each split and returns it alongside scores.
    fit_params : Optional[Dict[str, Any]], default None
        Additional keyword arguments to pass to the estimator's fit method.
    progress : bool, default True
        Progress bar: lazy tqdm.auto wrapping of the inner split loop. Defaults
        to True since builtins enable it for user-visible feedback; set False to
        silence (e.g. when running headless or composing engines programmatically).    
    progress_desc : Optional[str], default None
        Description for the progress bar.
    bypc : bool, default False
        DEPRECATED. Legacy combined flag that tangled the asymmetric resampling
        profile and the per-component output shape into one switch. Passing
        ``bypc=True`` is equivalent to ``mode='asym'`` (when ``mode`` is unset) and
        ``per_component=True``. Emits a ``DeprecationWarning``; will be removed in a
        future release. New code should set ``mode`` and ``per_component`` explicitly.

    Methods
    -------
    __call__(X: pd.DataFrame, y: Any, group: Optional[str] = None) -> Any
        Executes the resampling loop on the input data and returns reproducibility scores.
    """
    def __init__(
            self,
            estimator: Any,
            cv: Any,
            mode: Optional[str] = None,
            per_component: bool = False,
            shuffle: bool = False,
            pro_cong: bool = False,
            subspace: bool = False,
            fit_params: Optional[Dict[str, Any]] = None,
            progress: bool = True,
            progress_desc: Optional[str] = None,
            bypc: bool = False,
        ) -> None:
            # Legacy `bypc=True` selected the asymmetric resampling AND the per-component
            # output shape together. Translate it into the two independent flags and warn.
            if bypc:
                warnings.warn(
                    "BootstrapEngine(bypc=True) is deprecated and will be removed in a "
                    "future release. Pass mode='asym' (for the asymmetric resampling "
                    "profile) and per_component=True (for per-component output) "
                    "explicitly instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if mode is None:
                    mode = "asym"
                per_component = True

            self.estimator = estimator
            self.cv = cv
            self.mode = mode
            self.per_component = per_component
            self.shuffle = shuffle
            self.pro_cong = pro_cong
            self.subspace = subspace
            self.fit_params = fit_params or {}
            self.progress = progress
            self.progress_desc = progress_desc

    def _total_splits(self) -> int:
        """
        Expected number of (x1, x2) pairs the inner loop will iterate, given the
        current cv mode. Used for the progress bar's denominator so it can report
        proportion-complete; iteration still works correctly if the count is wrong.
        """
        if self.mode == "omnibus" or self.mode == "splithalf":
            return self.cv.n_redists
        # `split` and `asym_split` share the same fold-product shape.
        k = self.cv.n_splits
        if not self.cv.boot:
            return k * k
        # boot=True: sum_{r=1}^{k} C(k, r) = 2^k - 1 combinations on each side
        n_combos = (2 ** k) - 1
        return n_combos * n_combos

    def __call__(self, X: pd.DataFrame, y: Any, group: Optional[str] = None) -> Any:
        """Enables object instance to be called exactly like the original function."""

        # Handle Full-Mantel Shuffling if requested
        if self.shuffle:
            # # Fallback protects against omitted 'group' strings in orchestration calls
            # group_col = group or getattr(self.cv, 'group', None)
            X = pd.DataFrame(fullmantel(X))

        # Resampling-profile dispatch (independent of output shape). "bypc" accepted
        # as a back-compat synonym for "asym" so old engine instances configured by
        # third-party code keep working.
        if self.mode == "omnibus" or self.mode == "splithalf":
            splits = self.cv.resample_pairs(df=X, subset=y)
        elif self.mode == "asym" or self.mode == "bypc":
            splits = self.cv.asym_split(X, y)
        else:
            splits = self.cv.split(X, y)

        # Optional progress bar over the inner split loop (lazy tqdm.auto import;
        # no-ops gracefully if tqdm isn't installed).
        if self.progress:
            try:
                from tqdm.auto import tqdm
                splits = tqdm(splits, total=self._total_splits(),
                              desc=self.progress_desc, leave=False)
            except ImportError:
                pass

        # Obtains scores and optional phi / subspace values for each split
        scores, phis, subs = [], [], []
        for x1, x2 in splits:
            self.estimator.fit(x1, x2, **self.fit_params)
            preds = self.estimator.predict()
            corrs = np.corrcoef(preds[0], preds[1], rowvar=False)

            scores.append(self.estimator.hom_pairs(corrs))
            if self.pro_cong:
                phis.append(self.estimator.pro_cong())
            if self.subspace:
                subs.append(self.estimator.subspace_sim())

        # Output-shape dispatch (independent of resampling profile)
        if self.per_component:
            complist = list(map(list, zip(*scores)))
            result = [complist, list(map(list, zip(*phis)))] if self.pro_cong else complist
            if self.subspace:
                if not self.pro_cong:
                    result = [complist]
                # Subspace similarity is a whole-solution property, not per-component, so
                # collapse each split's per-direction cosines to one value reported per sample.
                result.append([float(np.mean(s)) for s in subs])
            return result

        # Return overall scores (and optional phis / subspace) as lists
        result = [scores, phis] if self.pro_cong else scores  # empty phis preserves index position
        if self.subspace:
            if not self.pro_cong:
                result = [scores]
            result.append(subs)
        return result
