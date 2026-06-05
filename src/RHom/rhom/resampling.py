from .._deps import pd, np, StandardScaler
from typing import Dict, List, Tuple, Generator, Optional, Union
from itertools import combinations, product

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

    Attributes
    ----------
        - scaler: StandardScaler
            - The scaler used for z-score normalization.

    Methods
    -------
        - standardize(df):
            - z-score normalizes an inputted dataframe using scaler.
    
        - assignModel(df, subrows):
            - Partitions rows of inputted dataset to standardized halves.

        - target_mask(df, mask, target_val):
            - Boolean row selector for the target subset, with branching logic for stratification and clustering.
        
        - stratified_target_mask(df):
            - Boolean mask selecting half of each stratum's rows, with cluster-aware logic when `cluster` is set.

        - cluster_partition(df, target_size):
            - Helper for target_mask: picks whole clusters (shuffled) until at least `target_size` rows are gathered.

        - to_features(data):
            - Returns decomposition columns as a numpy array, dropping any group / cluster / stratify labels.

        - make_folds(data):
            - Partitions rows into `n_splits` folds, with cluster-aware logic when `cluster` is set.

        - stratified_make_folds(data):
            - K folds with proportional sampling from each stratum, with cluster-aware logic when `cluster` is set.

        - omni_prep(df, subrows):
            - Prepares data for omnibus-sample reproducibility by partitioning and standardizing.

        - omni_prep_mini(df, subsamps, subset, subrows):
            - Prepares data for omnibus-sample reproducibility analysis with a specified subset.

        - split(X, y):
            - Used in direct-projection reproducibility. Splits the referent and comparate dataframes into folds.

        - bypc_split(X, y):
            - Used in by-component omnibus-sample reproducibility. Splits omnibus and sample sets into folds.

        - holdout_split(X):
            - Used in held-out cross-validation (i.e., LOGO, K-fold, stratified K-fold) reproducibility. 
              Yields train/test splits based on grouping or stratification.

        - redists(df, subset):
            - Bootstrap resamples the split-half or omnibus-sample subdivisions of an inputted dataframe.
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
        ) -> None:
            self.n_splits = k
            self.n_redists = n
            self.boot = boot
            self.omnibus = omnibus
            self.group = group
            self.cluster = cluster
            self.stratify = stratify
            self.stratified_kfold = stratified_kfold

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
        """Return decomposition columns as a numpy array, dropping any group / cluster / stratify labels."""
        if hasattr(data, 'columns'):
            drop = [c for c in (self.group, self.cluster, self.stratify) if c and c in data.columns]
            data = data.drop(labels=drop, axis=1) if drop else data
            return np.array(data.values, copy=True)
        return np.array(data, copy=True)

    def _make_folds(self, data: Union[pd.DataFrame, np.ndarray]) -> List[np.ndarray]:
        """
        Partition rows into `n_splits` folds.

        With `cluster` set (and present on a DataFrame), whole level-2 units are assigned
        to folds so a cluster never spans two folds (requires at least `n_splits` distinct
        clusters). Otherwise rows are shuffled and split directly.
        """
        if self.cluster and hasattr(data, 'columns') and self.cluster in data.columns:
            clusters = data[self.cluster].unique().copy()
            np.random.shuffle(clusters)
            labels = data[self.cluster].values
            feat = self._to_features(data)
            return [feat[np.isin(labels, fold)] for fold in np.array_split(clusters, self.n_splits)]

        arr = self._to_features(data)
        np.random.shuffle(arr)
        return np.array_split(arr, self.n_splits)

    def _stratified_make_folds(self, data: pd.DataFrame) -> List[np.ndarray]:
        """
        K folds with proportional sampling from each stratum.

        For each level of ``self.stratify`` the rows are partitioned into ``n_splits``
        folds via ``_make_folds`` (which is cluster-aware when ``self.cluster`` is set),
        and the i-th fold is the concatenation of the i-th sub-fold from every stratum.
        Returns a list of feature arrays (group / cluster / stratify columns stripped
        by ``_to_features`` inside ``_make_folds``).
        """
        if not (self.stratify and self.stratify in data.columns):
            raise ValueError("stratified_make_folds requires self.stratify to be set and present in data.")

        folds: List[List[np.ndarray]] = [[] for _ in range(self.n_splits)]
        for level in data[self.stratify].unique():
            level_data = data[data[self.stratify] == level]
            for i, sub in enumerate(self._make_folds(level_data)):
                if len(sub) > 0:
                    folds[i].append(sub)
        return [np.concatenate(parts, axis=0) if parts else np.empty((0, 0)) for parts in folds]

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

        models["omnibus"] = self.standardize(pd.concat(omnibus_chunks, axis=0))
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
                yield x1_c[fold[0]], x2_c[fold[1]]
        else:
            boot_combinations = []
            for z in range(1, self.n_splits + 1):
                boot_combinations.extend(list(combinations(foldidx, r=z)))
                
            boot_folds = list(product(boot_combinations, repeat=2))
            for z in boot_folds:
                f1 = np.concatenate([x1_c[v] for v in z[0]], axis=0)
                f2 = np.concatenate([x2_c[v] for v in z[1]], axis=0)
                yield f1, f2

    def bypc_split(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into referent and comparate sets based on grouping."""
        foldidx = list(range(self.n_splits))
        X_arr = self._to_features(X)
        x_c = self._make_folds(y)

        if not self.boot:
            for z in product(foldidx, repeat=2):
                yield X_arr, x_c[z[1]]
        else:
            boot_combinations = []
            for z in range(1, self.n_splits + 1):
                boot_combinations.extend(list(combinations(foldidx, r=z)))
                
            boot_folds = list(product(boot_combinations, repeat=2))
            for fold_pair in boot_folds:
                yield X_arr, np.concatenate([x_c[v] for v in fold_pair[1]], axis=0)

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
                train = np.concatenate([folds[j] for j in range(nf) if j != i], axis=0)
                yield train, folds[i], f"fold{i + 1}"
            return

        if self.group is not None and hasattr(X, "columns") and self.group in X.columns:
            drop = [c for c in (self.group, self.cluster, self.stratify) if c and c in X.columns]
            for g in X[self.group].unique():
                is_test = X[self.group] == g
                train = X[~is_test].drop(labels=drop, axis=1, errors='ignore').values
                test = X[is_test].drop(labels=drop, axis=1, errors='ignore').values
                yield train, test, str(g)
            return

        folds = self._make_folds(X)
        nf = len(folds)
        for i in range(nf):
            train = np.concatenate([folds[j] for j in range(nf) if j != i], axis=0)
            yield train, folds[i], f"fold{i + 1}"

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
                # Reuses the exact same partition engine for split-half analyses
                h1, h2 = self._assign_model(splitdf, mask, 1)
                yield [h1, h2]
    