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

        - omni_prep(df, subrows):
            - Prepares data for omnibus-sample reproducibility by partitioning and standardizing.

        - omni_prep_mini(df, subsamps, subset, subrows):
            - Prepares data for omnibus-sample reproducibility analysis with a specified subset.

        - split(X, y):
            - Used in direct-projection reproducibility. Splits the referent and comparate dataframes into folds.

        - bypc_split(X, y):
            - Used in by-component omnibus-sample reproducibility. Splits omnibus and sample sets into folds.

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
            cluster: Optional[str] = None
        ) -> None:
            self.n_splits = k
            self.n_redists = n
            self.boot = boot
            self.omnibus = omnibus
            self.group = group
            self.cluster = cluster

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

        drop = [c for c in (self.group, self.cluster) if c and c in df.columns]
        half1 = df[is_target].drop(labels=drop, axis=1, errors='ignore')
        half2 = df[~is_target].drop(labels=drop, axis=1, errors='ignore')

        return self.standardize(half1), self.standardize(half2)

    def _target_mask(self, df: pd.DataFrame, mask: np.ndarray, target_val: Union[str, int]) -> np.ndarray:
        """
        Boolean row selector for the target subset.

        With `cluster` set, whole level-2 units are selected together until the target row
        count is reached; otherwise the row mask is shuffled and applied directly.
        """
        if self.cluster and self.cluster in df.columns:
            target_size = int(np.sum(np.asarray(mask) == target_val))
            return df.index.isin(self._cluster_partition(df, target_size))

        mask = np.array(mask, copy=True)
        np.random.shuffle(mask)
        return np.asarray(mask) == target_val

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
        """Return decomposition columns as a numpy array, dropping any group/cluster labels."""
        if hasattr(data, 'columns'):
            drop = [c for c in (self.group, self.cluster) if c and c in data.columns]
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
    