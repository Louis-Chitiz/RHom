from .._deps import pd, np, StandardScaler
from typing import Dict, List, Tuple, Generator, Optional, Union
from itertools import combinations, product

from ..preprocessing.data_utils import group_standardize

class pair_cv():
    """
    Resampling Methods for Bootstrapping Component Reproducibility Analysis
    ------------------------------------------------------------------------
    A class for bootstrap resampling component-similarity analyses.

    Parameters
    ----------
        - k: int, default=5
            - Number of folds to cross-validate direct-projection reproducibility.
        - n: int, default=1000
            - Number of resamples to bootstrap similarity scores.
        - boot: bool, default=False
            - Whether to bootstrap direct-projection reproducibility by comparing every combination of folds in the referent dataset with every combination of folds in the comparate.
        - omnibus: bool, default=False
            - Whether to bootstrap resample according to omnibus-sample reproducibility or split-half reliability.
        - group: str, default=None
            - The name of the column to be the selection variable by which to conduct separate reproducibility analyses.
        - cluster: str, default=None
            - The name of a level-2 / clustering column (e.g. participant ID). When set, whole clusters
              are kept together in every split, fold, and bootstrap draw so a unit never appears on both
              sides of a comparison. Prevents leakage and pseudoreplication with nested data.
        - stratify: str, default=None
            - Stratification: when set, half-construction (splithalf) and k-fold partitioning
              (holdout) draw proportionally from each level of this column instead of from
              the global pool. Composes with `cluster` -- stratification happens first, and
              within each stratum clusters are kept intact.
        - stratified_kfold: bool, default=False
            - Toggle that lets holdout_split distinguish stratified k-fold from LOGO when
              both `group` and folds are meaningful (the holdout_cv builtin sets stratify
              the user's group column and flips this flag on).
        - groupby: str, default=None
            - Optional nuisance-grouping column for groupedPCA-style decomposition. When
              set, every materialized decomposition subset (half, fold, sample, omnibus)
              is z-scored *within each level of this column on its own rows* before the
              labels are stripped, so each resample reproduces groupedPCA's within-group
              standardization with no leakage across the split. Independent of `group`
              (the comparison/iteration variable), `stratify`, and `cluster`.

    Notes
    -----
    Each method carries its own docstring; the public entry points are ``split`` /
    ``asym_split`` (fold-cross profiles), ``holdout_split`` (train/test CV), ``redists``
    (split-half / omnibus bootstrap), and ``bootstrap_resamples`` (consensus). Every
    method materialises its decomposition subsets through the single ``_prep`` helper
    (plus ``_make_folds`` for fold-based protocols), which is where the cluster /
    stratify / groupby handling lives -- so behaviour changes belong there, not in the
    individual splitters.
        """

    def __init__(
            self,
            k: int = 5,
            n: int = 1000,
            boot: bool = False,
            omnibus: bool = False,
            group: Optional[str] = None,
            cluster: Optional[str] = None,
            stratify: Optional[str] = None,
            stratified_kfold: bool = False,
            groupby: Optional[str] = None,
        ) -> None:
            self.n_splits = k
            self.n_redists = n
            self.boot = boot
            self.omnibus = omnibus
            self.group = group
            self.cluster = cluster
            self.stratify = stratify
            self.stratified_kfold = stratified_kfold
            self.groupby = groupby

    @staticmethod
    def standardize(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the values in the DataFrame using z-score normalization."""
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    def _assign_model(self, df: pd.DataFrame, mask: np.ndarray, target_val: Union[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Consolidated helper to split a frame into two standardized halves.

        When `cluster` is set, whole level-2 units are kept together (a cluster is
        never split across the two halves); otherwise rows are split directly.
        Returns (target_half, remaining_half).
        """
        df = df.copy()
        is_target = self._target_mask(df, mask, target_val)

        drop = [c for c in (self.group, self.cluster, self.stratify) if c and c in df.columns]

        if self.groupby and self.groupby in df.columns:
            # Within-group standardize each half on its own rows (groupedPCA per
            # resample), then drop labels incl. groupby. Returned as DataFrames so
            # callers like omni_prep_mini can still concatenate before a final pass.
            gb_drop = drop + ([self.groupby] if self.groupby not in drop else [])
            half1 = group_standardize(df[is_target], self.groupby).drop(labels=gb_drop, axis=1, errors='ignore')
            half2 = group_standardize(df[~is_target], self.groupby).drop(labels=gb_drop, axis=1, errors='ignore')
            return half1, half2

        half1 = df[is_target].drop(labels=drop, axis=1, errors='ignore')
        half2 = df[~is_target].drop(labels=drop, axis=1, errors='ignore')

        return self.standardize(half1), self.standardize(half2)

    def _target_mask(self, df: pd.DataFrame, mask: np.ndarray, target_val: Union[str, int]) -> np.ndarray:
        """
        Boolean row selector for the target subset.

        Dispatch order:
            * ``self.stratify`` set -- pick ~half of each stratum's rows (cluster-aware
              within each stratum when ``self.cluster`` is also set). ``mask`` /
              ``target_val`` are ignored: the stratified case always produces a roughly
              50/50 split, with the complement returned as the other half.
            * ``self.cluster`` set -- whole clusters are taken together until the
              ``mask`` target count is met.
            * otherwise -- the row mask is shuffled globally and applied directly.
        """
        if self.stratify and self.stratify in df.columns:
            return self._stratified_target_mask(df)

        if self.cluster and self.cluster in df.columns:
            target_size = int(np.sum(np.asarray(mask) == target_val))
            return df.index.isin(self._cluster_partition(df, target_size))

        mask = np.array(mask, copy=True)
        np.random.shuffle(mask)
        return np.asarray(mask) == target_val

    def _stratified_target_mask(self, df: pd.DataFrame) -> np.ndarray:
        """
        Boolean mask selecting ``floor(n_stratum / 2)`` rows from each stratum.

        The complement (returned by ``_assign_model`` as the second half) holds the
        remaining ``ceil(n_stratum / 2)`` rows per stratum, so a roughly equal sample
        from every level lands on each side. With ``self.cluster`` set, whole clusters
        are kept together within each stratum.
        """
        chosen_indices: List = []
        for level in df[self.stratify].unique():
            level_df = df[df[self.stratify] == level]
            target_size = len(level_df) // 2
            if target_size == 0:
                continue
            if self.cluster and self.cluster in df.columns:
                level_chosen = self._cluster_partition(level_df, target_size)
            else:
                level_chosen = np.random.choice(level_df.index, size=target_size, replace=False)
            chosen_indices.extend(np.asarray(level_chosen).tolist())
        return df.index.isin(chosen_indices)

    def _cluster_partition(self, df: pd.DataFrame, target_size: int) -> np.ndarray:
        """Pick whole clusters (shuffled) until at least `target_size` rows are gathered."""
        clusters = df[self.cluster].unique().copy()
        np.random.shuffle(clusters)

        chosen, count = [], 0
        for c in clusters:
            rows = np.asarray(df.index[df[self.cluster] == c])
            chosen.append(rows)
            count += len(rows)
            if count >= target_size:
                break

        return np.concatenate(chosen) if chosen else np.asarray([])

    def _to_features(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return decomposition columns as a numpy array, dropping any group / cluster / stratify / groupby labels."""
        if hasattr(data, 'columns'):
            drop = [c for c in (self.group, self.cluster, self.stratify, self.groupby) if c and c in data.columns]
            data = data.drop(labels=drop, axis=1) if drop else data
            return np.array(data.values, copy=True)
        return np.array(data, copy=True)

    def _prep(self, data: Union[pd.DataFrame, np.ndarray], global_std: bool) -> np.ndarray:
        """
        Materialize a decomposition-ready feature array from a labeled row subset.

        This is the single chokepoint every resampling method routes its final
        decomposition units through, so groupedPCA support lives in one place:

            * ``self.groupby`` set -- z-score features *within each group* on this
              subset's own rows (leakage-free per-resample groupedPCA standardization).
              basePCA's later global scaling is then an exact identity, so the grouped
              solution falls out of the unchanged pipeline.
            * otherwise -- historical behaviour: global z-scoring when ``global_std``
              (the splithalf / omnibus regime, where the resampler standardized) or raw
              features otherwise (the fold regime, where basePCA does the scaling).

        All label columns are stripped from the result either way.
        """
        if self.groupby and hasattr(data, "columns") and self.groupby in data.columns:
            return self._to_features(group_standardize(data, self.groupby))

        feats = self._to_features(data)
        if global_std:
            return self.standardize(pd.DataFrame(feats)).to_numpy()
        return feats

    def _make_folds(self, data: Union[pd.DataFrame, np.ndarray]) -> List[pd.DataFrame]:
        """
        Partition rows into `n_splits` folds, returned as labeled DataFrame subsets.

        With `cluster` set, whole level-2 units are assigned to folds so a cluster never
        spans two folds (requires at least `n_splits` distinct clusters). Otherwise rows
        are shuffled and split directly.

        Standardization is *deferred*: folds keep their label columns and raw values, and
        the callers route each assembled decomposition unit (a single fold, a multi-fold
        train set, a stratified fold) through ``_prep``. This way a train set or a
        stratified fold is standardized as a whole rather than per sub-fold -- essential
        for leakage-free per-resample group standardization.
        """
        if not hasattr(data, 'columns'):
            data = pd.DataFrame(data)

        if self.cluster and self.cluster in data.columns:
            clusters = data[self.cluster].unique().copy()
            np.random.shuffle(clusters)
            return [data[data[self.cluster].isin(fold)] for fold in np.array_split(clusters, self.n_splits)]

        # Positional shuffle keeps this robust to duplicate index labels.
        pos = np.arange(len(data))
        np.random.shuffle(pos)
        return [data.iloc[chunk] for chunk in np.array_split(pos, self.n_splits)]

    def _stratified_make_folds(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        K folds with proportional sampling from each stratum.

        For each level of ``self.stratify`` the rows are partitioned into ``n_splits``
        folds via ``_make_folds`` (cluster-aware when ``self.cluster`` is set), and the
        i-th fold is the concatenation of the i-th sub-fold from every stratum. Returns
        labeled DataFrame folds (standardization deferred to the caller's ``_prep``).
        """
        if not (self.stratify and self.stratify in data.columns):
            raise ValueError("stratified_make_folds requires self.stratify to be set and present in data.")

        folds: List[List[pd.DataFrame]] = [[] for _ in range(self.n_splits)]
        for level in data[self.stratify].unique():
            level_data = data[data[self.stratify] == level]
            for i, sub in enumerate(self._make_folds(level_data)):
                if len(sub) > 0:
                    folds[i].append(sub)
        return [pd.concat(parts, axis=0) if parts else data.iloc[:0] for parts in folds]

    def bootstrap_resamples(self, X: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """
        Yield ``self.n_redists`` bootstrap resamples of ``X``.

        Mode toggle (uses the same ``cluster`` / ``stratify`` conventions as the rest
        of ``pair_cv``):
            * ``self.cluster`` set on X -- sample whole clusters with replacement so
              level-2 unit structure is preserved (bootstrap over clusters, not rows).
            * ``self.stratify`` set on X -- within each level of the stratifier,
              resample rows with replacement; concatenate across levels so every
              stratum keeps its original sample size.
            * neither -- plain row-wise bootstrap with replacement.

        Yields DataFrames (not numpy arrays) so feature-column metadata survives
        for downstream PCA fits; group / cluster / stratify columns are NOT dropped
        here -- the caller selects the decomposition columns from the result.
        """
        for _ in range(self.n_redists):
            if self.cluster and self.cluster in X.columns:
                clusters = X[self.cluster].unique()
                chosen = np.random.choice(clusters, size=len(clusters), replace=True)
                yield pd.concat([X[X[self.cluster] == c] for c in chosen])
            elif self.stratify and self.stratify in X.columns:
                chunks = []
                for level in X[self.stratify].unique():
                    level_rows = X[X[self.stratify] == level]
                    idx = np.random.choice(level_rows.index, size=len(level_rows), replace=True)
                    chunks.append(X.loc[idx])
                yield pd.concat(chunks)
            else:
                idx = np.random.randint(0, len(X), size=len(X))
                yield X.iloc[idx]

    def omni_prep(self, df: pd.DataFrame, subrows: Optional[Union[int, float]] = None) -> Dict[str, pd.DataFrame]:
        """
        Prepares data for by-component omnibus-sample reproducibility.

        Returns a dict keyed by each group value (its sample subset) plus "omnibus"
        (the standardized aggregate of every group's omnibus portion). The per-sample
        frames keep the cluster column and stay unstandardized so the downstream CV folds
        in `bypc_split` remain cluster-aware; the estimator standardizes each fold itself.
        """
        if self.group is None:
            raise ValueError("Group column must be specified for omnibus preprocessing.")

        samples = df[self.group].unique()
        models: Dict[str, pd.DataFrame] = {}
        omnibus_chunks = []

        for sample in samples:
            subsamp_df = df[df[self.group] == sample].copy()

            mask = np.full(subsamp_df.shape[0], "omnibus", dtype=object)
            if subrows is not None:
                mask[:int(subrows)] = "sample"

            is_sample = self._target_mask(subsamp_df, mask, "sample")

            # Sample side: drop only the group label, keep the cluster column for fold-level CV.
            models[sample] = subsamp_df[is_sample].drop(labels=self.group, axis=1, errors='ignore')

            # Omnibus side: decomposed whole, so make it feature-only (group + cluster removed).
            omni_drop = [c for c in (self.group, self.cluster) if c and c in subsamp_df.columns]
            omnibus_chunks.append(subsamp_df[~is_sample].drop(labels=omni_drop, axis=1, errors='ignore'))

        # The omnibus is a single fixed reference set, so standardize it as one unit
        # (within-group when groupby is set, else global). omni_drop above keeps the
        # groupby column on each chunk so _prep can see it here.
        models["omnibus"] = self._prep(pd.concat(omnibus_chunks, axis=0), global_std=True)
        return models

    def omni_prep_mini(
        self, 
        df: pd.DataFrame, 
        subsamps: Dict[str, pd.DataFrame], 
        subset: str, 
        subrows: Optional[Union[int, float]] = None
    ) -> List[pd.DataFrame]:
        """
        Prepare data for omnibus-sample reproducibility 
        analysis with a specified subset.
        """
        # Process primary subset
        target_df = df[df[self.group] == subset]
        mask = np.full(target_df.shape[0], "omnibus", dtype=object)
        if subrows is not None:
            mask[:int(subrows)] = "sample"
            
        sample_subset, target_omni = self._assign_model(target_df, mask, "sample")
        omnibus_chunks = [target_omni]

        # Process complementary sets
        for _, subsamp_df in subsamps.items():
            sub_mask = np.full(subsamp_df.shape[0], "omnibus", dtype=object)
            if subrows is not None:
                sub_mask[:int(subrows)] = "sample"
            _, other_omni = self._assign_model(subsamp_df, sub_mask, "sample")
            omnibus_chunks.append(other_omni)

        omnibus_df = self.standardize(pd.concat(omnibus_chunks, axis=0))
        return [sample_subset, omnibus_df]

    def split(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into referent and comparate sets."""
        foldidx = list(range(self.n_splits))
        x1_c = self._make_folds(X)
        x2_c = self._make_folds(y)

        if not self.boot:
            for fold in product(foldidx, repeat=2):
                yield (self._prep(x1_c[fold[0]], global_std=False),
                       self._prep(x2_c[fold[1]], global_std=False))
        else:
            boot_combinations = []
            for z in range(1, self.n_splits + 1):
                boot_combinations.extend(list(combinations(foldidx, r=z)))

            boot_folds = list(product(boot_combinations, repeat=2))
            for z in boot_folds:
                f1 = pd.concat([x1_c[v] for v in z[0]], axis=0)
                f2 = pd.concat([x2_c[v] for v in z[1]], axis=0)
                yield self._prep(f1, global_std=False), self._prep(f2, global_std=False)

    def asym_split(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Asymmetric split: X is taken whole on every iteration; y is partitioned into
        ``n_splits`` folds and (under ``boot=True``) every non-empty combination of
        those folds is iterated. Pairs of ``(X_whole, y_fold_combo)`` are yielded.

        This is the resampling profile used by ``omsamp_bypc`` (where X is the
        fixed omnibus side and y is the sample side being folded). It is decoupled
        from the per-component output shape -- callers requesting per-component
        scores should pass ``per_component=True`` to the engine, which works with
        any resampling profile, not just this one. The name reflects the asymmetry
        between the two sides (X-fixed vs y-folded). ``bypc_split`` is kept as a back-compat
        alias for one release.
        """
        foldidx = list(range(self.n_splits))
        # X is the fixed side, taken whole; prep it once. (Under groupby it is
        # standardized as a single unit -- it is the reference, not a resample.)
        X_arr = self._prep(X, global_std=False)
        x_c = self._make_folds(y)

        if not self.boot:
            for z in product(foldidx, repeat=2):
                yield X_arr, self._prep(x_c[z[1]], global_std=False)
        else:
            boot_combinations = []
            for z in range(1, self.n_splits + 1):
                boot_combinations.extend(list(combinations(foldidx, r=z)))

            boot_folds = list(product(boot_combinations, repeat=2))
            for fold_pair in boot_folds:
                yield X_arr, self._prep(pd.concat([x_c[v] for v in fold_pair[1]], axis=0), global_std=False)

    # Back-compat alias -- the method was named bypc_split historically because
    # ``omsamp_bypc`` was the only analysis that used it. New code should call
    # ``asym_split`` directly.
    bypc_split = asym_split

    def holdout_split(self, X: Union[pd.DataFrame, np.ndarray]) -> Generator[Tuple[np.ndarray, np.ndarray, str], None, None]:
        """
        Held-out cross-validation splits: each iteration yields ``(train, test, label)``.

        Mode toggle (checked in order):
            * ``self.stratified_kfold`` AND ``self.stratify`` set on X --
              stratified K-fold: each fold contains proportional rows from every level
              of ``self.stratify``. ``label`` is ``"fold{i}"``.
            * ``self.group`` set and present on X -- leave-one-group-out: each level
              of ``self.group`` held out as the test set once. ``label`` is the group's
              name.
            * otherwise -- random K-fold partition (cluster-aware when ``self.cluster``
              is set on a DataFrame, via the same ``_make_folds`` logic used by
              ``split`` / ``bypc_split``). ``label`` is ``"fold{i}"``.

        Standardisation is NOT applied here -- the estimator (``rhom`` -> ``basePCA``)
        standardises each fitted side internally, so train and test are scaled against
        their own statistics rather than a shared one. This matches the behaviour of
        ``split`` and ``bypc_split``.
        """
        if (self.stratified_kfold and self.stratify is not None
                and hasattr(X, "columns") and self.stratify in X.columns):
            folds = self._stratified_make_folds(X)
            nf = len(folds)
            for i in range(nf):
                train = pd.concat([folds[j] for j in range(nf) if j != i], axis=0)
                yield (self._prep(train, global_std=False),
                       self._prep(folds[i], global_std=False), f"fold{i + 1}")
            return

        if self.group is not None and hasattr(X, "columns") and self.group in X.columns:
            for g in X[self.group].unique():
                is_test = X[self.group] == g
                yield (self._prep(X[~is_test], global_std=False),
                       self._prep(X[is_test], global_std=False), str(g))
            return

        folds = self._make_folds(X)
        nf = len(folds)
        for i in range(nf):
            train = pd.concat([folds[j] for j in range(nf) if j != i], axis=0)
            yield (self._prep(train, global_std=False),
                   self._prep(folds[i], global_std=False), f"fold{i + 1}")

    def redists(self, df: pd.DataFrame, subset: Optional[str] = None) -> Generator[List[pd.DataFrame], None, None]:
        """
        Generates a set of bootstrap reassignments of different subdivisions lazily.
        
        Yields
        ------
        List[pd.DataFrame]
            A list containing the pair of split dataframes for a single bootstrap slice.
        """
        if self.omnibus:
            if self.group is None:
                raise ValueError("Group column must be specified when omnibus=True.")
                
            samples = df[self.group].unique()
            subsamps = {sample: df[df[self.group] == sample] for sample in samples if sample != subset}

            nrows = df[self.group].value_counts()
            nval = nrows.min() / 2

            for _ in range(self.n_redists):
                # Streams memory safely by yielding on the fly
                yield self.omni_prep_mini(df=df, subset=subset, subsamps=subsamps, subrows=nval)
                
        else:
            splitdf = df[df[self.group] == subset].drop(labels=self.group, axis=1) if (self.group is not None and subset is not None) else df.copy()

            rows = splitdf.shape[0]
            mask = np.full(rows, 2)
            mask[:int(rows / 2)] = 1

            for _ in range(self.n_redists):
                # Same partition engine as _assign_model, but each half is routed through
                # _prep so it is the final decomposition unit (global-standardized, or
                # within-group standardized when groupby is set).
                is_target = self._target_mask(splitdf, mask, 1)
                yield [self._prep(splitdf[is_target], global_std=True),
                       self._prep(splitdf[~is_target], global_std=True)]
    