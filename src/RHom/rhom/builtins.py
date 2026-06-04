from .._deps import pd, np, randint, plt, StandardScaler

import os
import copy
import warnings

from itertools import combinations

from ..core.base_pca import basePCA
from ..preprocessing.preliminary import _check_rank
from ..io.save import setupanalysis
from ..visualization.wordclouds import save_wordclouds

from .metrics import rhom
from .resampling import pair_cv
from .bootstrap import BootstrapEngine


def _summary_stats(distribution, alpha=0.05):
    """Internal statistical helper to calculate confidence intervals."""
    distribution = np.asarray(distribution)
    mean = np.mean(distribution)
    std_err = np.std(distribution, ddof=1) if len(distribution) > 1 else 0.0
    
    # Calculate a simple normal distribution or percentile-based CI
    low_ci = np.percentile(distribution, (alpha / 2) * 100)
    high_ci = np.percentile(distribution, (1 - alpha / 2) * 100)
    
    return {
        "x": mean,
        "se": std_err,
        "LCI": low_ci,
        "UCI": high_ci
    }

def _build_row(n_comp, rhm_data, phi_data=None, sub_data=None, metadata=None):
    """Assembles a single unified row mapping for reporting metrics."""
    row = {"n_comp": f"{n_comp}PC" if isinstance(n_comp, int) else n_comp}
    if metadata:
        row.update(metadata)

    rhm_stats = _summary_stats(rhm_data)
    row.update({f"rhm_{k}": v for k, v in rhm_stats.items()})

    if phi_data is not None:
        phi_stats = _summary_stats(phi_data)
        row.update({f"phi_{k}": v for k, v in phi_stats.items()})

    if sub_data is not None:
        sub_stats = _summary_stats(sub_data)
        row.update({f"sub_{k}": v for k, v in sub_stats.items()})

    return row

def _display_stats(header, rhm_stats, phi_stats):
    print(f"{header}:\n" + "*"*20)
    print(f"Mean Homologue Similarity: {rhm_stats['x']:.3g} +/- {rhm_stats['se']:.3g} 95% CI[{rhm_stats['LCI']:.3g}, {rhm_stats['UCI']:.3g}]")
    if phi_stats:
        print(f"Mean Factor Congruence:    {phi_stats['x']:.3g} +/- {phi_stats['se']:.3g} 95% CI[{phi_stats['LCI']:.3g}, {phi_stats['UCI']:.3g}]")
    print("*"*40)

def _export_report(df, path, prefix, suffix):
    """Handles directory confirmation and saves the summary CSV file safely."""
    setupanalysis(path, prefix, includetime = False)
    filename = f"{prefix}_{suffix}.csv"
    full_path = os.path.join(path, f"{prefix}", filename)
    df.to_csv(full_path, index=False)
    print(f"Dataframe saved to: {full_path}")

def splithalf(df=None, group=None, npc=None, method='svd', rotation='varimax', corr='pearson',
              boot=1000, save=True, display=False, shuffle=False, cluster=None, stratify=None,
              subspace=False, path='results', file_prefix=randint(10000, 99999)):
    """
    Split-Half Reliability
    ----------------------
    This function conducts a bootstrapped split-half reliability analysis
    on your dataframe. It can do so on a full dataset, or at each level of a
    grouping variable. It simply bootstrap reassigns random halves of the data
    into two subsets and computes their component similarity based on:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)

    Parameters
    ----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        cluster: str, default=None
            Optional level-2 / clustering column (e.g. participant ID). When provided,
            whole clusters are kept together in every resample, fold, and split, so a unit
            never appears on both sides of a comparison. Prevents leakage and
            pseudoreplication with nested data. Requires at least `folds` distinct clusters
            per group where cross-validation is used.

        stratify: str, default=None
            Optional stratification column for whole-dataset splithalf. When provided
            (and ``group`` is None), each bootstrap half is drawn proportionally from
            every level of this column so a small source can't be over-represented in
            one half. Mutually exclusive with ``group`` (which means "iterate per level"
            rather than "balance across levels"). Composes with ``cluster``: within each
            stratum, whole clusters are kept on one side.

        subspace: bool, default=False
            If True, also report subspace similarity via principal angles between the two
            loading subspaces (sub_* columns; rotation- and order-invariant, in [0, 1]).

        corr: str, default="pearson"
            Which correlation matrix to decompose under `method='eigen'`: "pearson",
            "spearman" (rank correlation, ordinal-friendly), or "polychoric" (latent
            correlation behind ordinal items via Olsson 1979 MLE; meaningful only for
            genuinely ordinal data and noticeably slower). Ignored under `method='svd'`,
            which is Pearson-only; pass `method='eigen'` to switch correlation type.

        npc: int, default=None
            Number of components to extract per solution.

        rotation: str, default="varimax"
            Rotation method to be performed on referent. "none" for no rotation.

        boot: int, default=1000
            Number of bootstrap samples to generate 95% confidence intervals.

        save: bool, default=True
            Save outputted split-half reliability to .csv.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        path: str, default='results'
            The path to the output directory.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns
    -------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.

        .csv:
            If save=True, will save /results to a csv.

        printed results:
            If display=True, prints the output directly in the terminal.
    """

    if stratify is not None and group is not None:
        raise ValueError(
            "Pass either group= (per-level iteration) or stratify= (balanced sampling "
            "from each level across whole-dataset halves), not both."
        )

    drop_cols = [c for c in (group, cluster, stratify) if c is not None]
    df_t = df.drop(labels=drop_cols, axis=1) if drop_cols else df
    samples = df[group].unique() if group else ['fulldata']
    _check_rank(df_t)

    boot_model = rhom(rd=copy.deepcopy(df_t.values), n_comp=npc,
                      method=method, rotation=rotation, corr=corr)
    cv = pair_cv(group=group, cluster=cluster, stratify=stratify, n=boot)

    boot_engine = BootstrapEngine(
        estimator=boot_model,
        cv=cv,
        splithalf=True,
        pro_cong=True,
        shuffle=shuffle,
        subspace=subspace
    )

    rows = []
    for sample in samples:
        print(f"Running Split-Half: {sample}")
        # Execute via the unified __call__ interface
        results = boot_engine(X=df, y=sample, group=group)

        meta = {group: sample} if group else {"Group": "fulldata"}
        row = _build_row(boot_model.n_comp, results[0], results[1],
                         sub_data=results[2] if subspace else None, metadata=meta)
        rows.append(row)

        if display:
            _display_stats(f"Split-Half Reliability for {sample}", row)

    split_df = pd.DataFrame(rows)
    if save:
        _export_report(split_df, path, file_prefix, f"splithalf_{len(df_t.columns)}D_{npc}PC")

    return split_df

def splithalf_bypc(df=None, group=None, npc=None, method='svd', rotation='varimax', corr='pearson',
                   boot=1000, save=True, plot=True, display=False, shuffle=False, cluster=None,
                   stratify=None, subspace=False,
                   path='results', file_prefix=randint(10000, 99999)):
    """
    Split-Half Reliability: By-Component
    ------------------------------------
    Bootstrapped split-half reliability with a per-component breakdown. For each
    sample (the whole dataset when ``group=None``, or each level of ``group``), an
    anchor PCA is fit on the sample's full data to define the canonical PC1..PC{npc}
    frame; every per-half PCA inside the bootstrap is Procrustes-aligned to that
    anchor (via ``rhom``'s ``anchor=`` parameter), so the column index k carries a
    consistent homologue identity across all replicates. The result is one row per
    (sample, comp) triple with rhm / phi (and optionally sub) CIs computed across
    bootstrap replicates.

    Same anchor-and-transpose architecture as ``dir_proj_bypc``, with the splithalf
    resampling protocol substituted for dir_proj's pairwise CV.

    Parameters
    ----------

        df: pd.Dataframe, default=None
            Decomposition columns plus the grouping / clustering / stratification
            columns if used.

        group: str, default=None
            Column heading for per-level iteration. With ``group=None`` the analysis
            runs on the whole dataset as one sample.

        cluster: str, default=None
            Optional level-2 / clustering column. Whole clusters are kept on one side
            of every bootstrap split.

        stratify: str, default=None
            Optional stratification column for whole-dataset splithalf. When provided
            (and ``group`` is None), each bootstrap half is drawn proportionally from
            every level. Mutually exclusive with ``group``.

        subspace: bool, default=False
            If True, also report subspace similarity via principal angles. Subspace
            similarity is a whole-solution property, so per-replicate values are
            broadcast across the npc rows for that sample (same convention as
            ``omsamp_bypc`` / ``dir_proj_bypc``).

        corr: str, default="pearson"
            Correlation matrix for ``method='eigen'``. See ``splithalf`` for options.

        npc: int, default=None
            Number of components to extract per solution.

        rotation: str, default="varimax"
            Rotation applied to anchor and per-half PCAs.

        boot: int, default=1000
            Number of bootstrap halves to draw per sample.

        shuffle: bool, default=False
            If True, Mantel-shuffle the feature columns *once at the top* and use the
            shuffled frame for both the anchor fit and the bootstrap halves. Diverges
            from ``splithalf``'s per-call shuffle so the anchor and the halves share
            the same null realisation (otherwise the alignment scores compare random
            halves against a real-structure anchor, which isn't a coherent null).

        save / plot / display / path / file_prefix:
            Same conventions as the other builtins.

    Returns
    -------
        pd.DataFrame:
            One row per (sample, comp) triple with rhm_*, phi_*, and optional sub_*
            summary columns. Anchor loadings for the first sample are attached via
            ``df.attrs["loadings"]`` so ``plot_bypc`` can render wordclouds without a
            disk round-trip.

        .csv:
            If save=True.

        .png:
            If plot=True, one ``plot_bypc`` figure per metric.
    """
    if stratify is not None and group is not None:
        raise ValueError(
            "Pass either group= (per-level iteration) or stratify= (balanced sampling "
            "from each level across whole-dataset halves), not both."
        )

    drop_cols = [c for c in (group, cluster, stratify) if c is not None]
    df_t = df.drop(labels=drop_cols, axis=1) if drop_cols else df
    samples = df[group].unique() if group else ['fulldata']
    _check_rank(df_t)

    # One-shot Mantel shuffle so anchor and halves share the same null realisation.
    # Pass only the feature columns so fullmantel doesn't accidentally treat a numeric
    # cluster ID as a feature; reassign by .values to overwrite in place.
    df_input = df.copy()
    if shuffle:
        from ..preprocessing.data_utils import fullmantel
        feat_only = df_input.drop(labels=drop_cols, axis=1, errors='ignore') if drop_cols else df_input
        df_input[feat_only.columns] = fullmantel(feat_only).values

    cv = pair_cv(group=group, cluster=cluster, stratify=stratify, n=boot)

    rows = []
    anchor_loadings_by_sample = {}

    for sample in samples:
        print(f"Running By-Component Split-Half: {sample}")

        # Anchor PCA on this sample's full data (the whole dataset when group is None,
        # or the sample's rows when iterating per group level).
        if group:
            sample_data = df_input[df_input[group] == sample].drop(
                labels=drop_cols, axis=1, errors='ignore'
            )
        else:
            sample_data = df_input.drop(labels=drop_cols, axis=1, errors='ignore') if drop_cols else df_input

        anchor_pca = basePCA(n_components=npc, rotation=rotation, method=method, corr=corr)
        anchor_pca.fit(sample_data)
        anchor_loadings_by_sample[sample] = anchor_pca.loadings.copy()

        boot_model = rhom(rd=copy.deepcopy(df_t.values), bypc=True, n_comp=npc,
                          method=method, rotation=rotation, corr=corr,
                          anchor=anchor_pca.loadings.to_numpy())

        # engine.bypc=False so cv.redists is used (splithalf path); estimator.bypc=True
        # plus the anchor makes hom_pairs / pro_cong / subspace_sim return per-component
        # lists per replicate that we transpose at the bottom. engine.shuffle=False
        # because we've already shuffled df_input once above.
        boot_engine = BootstrapEngine(
            estimator=boot_model,
            cv=cv,
            splithalf=True,
            pro_cong=True,
            shuffle=False,
            subspace=subspace,
        )

        results = boot_engine(X=df_input, y=sample, group=group)
        # results layout with estimator.bypc=True, engine.bypc=False:
        #   results[0]: list-of-lists [n_replicates × npc]  -- |r| per component per replicate
        #   results[1]: list-of-lists [n_replicates × npc]  -- TCC per component per replicate
        #   results[2]: list-of-lists [n_replicates × npc]  -- subspace cosines (if subspace=True)
        rhm_per_comp = list(map(list, zip(*results[0])))   # [npc × n_replicates]
        phi_per_comp = list(map(list, zip(*results[1])))

        # Subspace similarity is a whole-solution property -- collapse each replicate's
        # per-direction cosines to one mean and broadcast across this sample's npc rows.
        sub_per_sample = ([float(np.mean(s)) for s in results[2]] if subspace else None)

        for idx in range(npc):
            meta = ({group: sample} if group else {"Group": "fulldata"})
            meta["comp"] = idx + 1
            row = _build_row(boot_model.n_comp, rhm_per_comp[idx], phi_per_comp[idx],
                             sub_data=sub_per_sample, metadata=meta)
            rows.append(row)

            if display:
                _display_stats(f"Split-Half: {sample} - Component {idx + 1}", row)

    splithalf_bypc_df = pd.DataFrame(rows)

    if plot:
        # Attach anchor loadings so plot_bypc can render wordclouds. When group is set
        # each sample has its own anchor; we attach the first sample's by default --
        # users can pass their own via plot_bypc(..., loadings=anchor_loadings_by_sample[<g>]).
        first_sample = samples[0]
        splithalf_bypc_df.attrs["loadings"] = anchor_loadings_by_sample[first_sample]

        from ..visualization.rhomplots import plot_bypc
        setupanalysis(path, file_prefix, includetime=False)
        plt.close('all')

        metrics = ["rhm", "phi"] + (["sub"] if subspace else [])
        for name in metrics:
            fig = plot_bypc(splithalf_bypc_df, metric=name)
            fig.savefig(
                os.path.join(path, f"{file_prefix}/{file_prefix}_splithalf_bypc_{len(df_t.columns)}D_{npc}PC_{name}.png"),
                bbox_inches="tight", dpi=150,
            )
            plt.show()
            plt.close(fig)

        # Persist the first sample's anchor loadings alongside the CSV
        loadings_path = os.path.join(
            path, f"{file_prefix}",
            f"{file_prefix}_loadings_{len(df_t.columns)}D_{npc}PC.csv",
        )
        anchor_loadings_by_sample[first_sample].to_csv(loadings_path)

    if save:
        _export_report(splithalf_bypc_df, path, file_prefix,
                       f"splithalf_bypc_{len(df_t.columns)}D_{npc}PC")

    return splithalf_bypc_df


def dir_proj(df=None, group=None, npc=None, method='svd', rotation="varimax", corr='pearson',
             folds=5, save=True, plot=True, display=False, shuffle=False, cluster=None,
             subspace=False, path='results', file_prefix=randint(10000, 99999)):
    """
    Direct-Projection Reproducibility
    ---------------------------------
    This function conducts a bootstrapped direct-projection analysis
    on your data based on some inputted grouping variable. This involves
    dividing each group into its own dataframe, and assessing the similarity
    of the components generated by each group to each other group based on:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)
    
    Parameters
    ----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        cluster: str, default=None
            Optional level-2 / clustering column (e.g. participant ID). When provided,
            whole clusters are kept together in every resample, fold, and split, so a unit
            never appears on both sides of a comparison. Prevents leakage and
            pseudoreplication with nested data. Requires at least `folds` distinct clusters
            per group where cross-validation is used.

        subspace: bool, default=False
            If True, also report subspace similarity via principal angles between the two
            loading subspaces (sub_* columns; rotation- and order-invariant, in [0, 1]).

        corr: str, default="pearson"
            Which correlation matrix to decompose under `method='eigen'`: "pearson",
            "spearman" (rank correlation, ordinal-friendly), or "polychoric" (latent
            correlation behind ordinal items via Olsson 1979 MLE; meaningful only for
            genuinely ordinal data and noticeably slower). Ignored under `method='svd'`,
            which is Pearson-only; pass `method='eigen'` to switch correlation type.

        npc: int, default=None
            Number of components to extract per solution.
        
        rotation: str, default="varimax"
            Rotation method to be performed on referent. "none" for no rotation.

        folds: int, default=5
            Number of folds to use for cross-validation.
        
        save: bool, default=True
            Save outputted reproducibility results to .csv.

        plot: bool, default=True
            Visualise results with heatmaps.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        path: str, default='results'
            The path to the output directory.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns
    -------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.
    
        .csv:
            If save=True, will save a .csv file to /results.

        .png:
            If plot=True, will save heatmaps for loading similarity and component-score similarity.

        printed results:
            If display=True, prints the output directly in the terminal.
    """
    
    cl = [cluster] if cluster else []
    groups = df[group].unique()
    # maindict retains the cluster column so folds can keep whole units together; it is
    # stripped from the decomposition inside pair_cv._make_folds.
    maindict = {g: df[df[group] == g].drop(labels=group, axis=1) for g in groups}
    pairings = list(combinations(groups, 2))

    scaler = StandardScaler()
    feat_cols = df.columns.drop([group, *cl])
    df_scaled = pd.DataFrame(scaler.fit_transform(df[feat_cols]), columns=feat_cols)
    _check_rank(df_scaled)

    boot_model = rhom(rd=copy.deepcopy(df_scaled.values), n_comp=npc,
                      method=method, rotation=rotation, corr=corr)
    cv = pair_cv(boot=True, k=folds, cluster=cluster)
    
    boot_engine = BootstrapEngine(
        estimator=boot_model,
        cv=cv,
        pro_cong=True,
        shuffle=shuffle,
        subspace=subspace
    )

    rows = []
    for ref, comp in pairings:
        print(f"Running {ref} x {comp}")

        _check_rank(maindict[ref].drop(labels=cl, axis=1))
        _check_rank(maindict[comp].drop(labels=cl, axis=1))
        # Execute using referent and comparator subsets
        results = boot_engine(X=maindict[ref], y=maindict[comp], group=group)

        meta = {'referent': ref, 'comparator': comp}
        row = _build_row(boot_model.n_comp, results[0], results[1],
                         sub_data=results[2] if subspace else None, metadata=meta)
        rows.append(row)

        if display:
            _display_stats(f"Direct Projection: {ref} x {comp}", row)

    dirproj_df = pd.DataFrame(rows)

    if plot:
        from ..visualization.rhomplots import plot_dirproj
        setupanalysis(path, file_prefix, includetime=False)
        plt.close('all')

        metrics = ["rhm", "phi"] + (["sub"] if subspace else [])
        for name in metrics:
            fig = plot_dirproj(dirproj_df, metric=name)
            fig.savefig(
                os.path.join(path, f"{file_prefix}/{file_prefix}_heatmap{len(df_scaled.columns)}D_{npc}PC_{name}.png"),
                bbox_inches="tight",
                dpi=150,
            )
            plt.show()
            plt.close(fig)

    if save:
        _export_report(dirproj_df, path, file_prefix, f"dj{len(df_scaled.columns)}D_{npc}PC")
        
    return dirproj_df

def dir_proj_bypc(df=None, group=None, npc=None, method='svd', rotation="varimax", corr='pearson',
                  folds=5, save=True, plot=True, display=False, shuffle=False, cluster=None,
                  subspace=False, path='results', file_prefix=randint(10000, 99999)):
    """
    Direct-Projection Reproducibility: By-Component
    -----------------------------------------------
    Pairwise direct-projection reproducibility broken down per component. A pooled-data
    PCA on the full dataset defines a canonical PC1..PC{npc} frame; every per-fold PCA
    in each pair is Procrustes-aligned to that anchor before scoring, so column index k
    refers to the same homologue across all (referent, comparator, replicate) triples.
    This is the bootstrapped analogue of the one-shot bypc heatmaps in the example
    script, and the dir_proj counterpart of `bypc` (which anchors against an omnibus
    half-sample rather than a pooled fit).

    Reports component similarity with:
        1) Loading similarity (Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (R-homologue: Mulholland et al., 2023; See also Everett, 1983)

    Parameters
    ----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        cluster: str, default=None
            Optional level-2 / clustering column (e.g. participant ID). When provided,
            whole clusters are kept together in every resample, fold, and split, so a unit
            never appears on both sides of a comparison. Prevents leakage and
            pseudoreplication with nested data. Requires at least `folds` distinct clusters
            per group.

        subspace: bool, default=False
            If True, also report subspace similarity via principal angles between the two
            loading subspaces. Subspace similarity is a whole-solution property, so the
            per-pair value is broadcast across the npc rows for that pair.

        corr: str, default="pearson"
            Which correlation matrix to decompose under `method='eigen'`: "pearson",
            "spearman" (rank correlation, ordinal-friendly), or "polychoric" (latent
            correlation behind ordinal items via Olsson 1979 MLE; meaningful only for
            genuinely ordinal data and noticeably slower). Ignored under `method='svd'`,
            which is Pearson-only; pass `method='eigen'` to switch correlation type.

        npc: int, default=None
            Number of components to extract per solution.

        rotation: str, default="varimax"
            Rotation method applied to the anchor PCA. "none" for no rotation.

        folds: int, default=5
            Number of folds to use for cross-validation within each pair.

        save: bool, default=True
            Save outputted reproducibility results to .csv.

        plot: bool, default=True
            Visualise results as a grid of per-component pairwise heatmaps.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        path: str, default='results'
            The path to the output directory.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns
    -------
        pd.DataFrame:
            One row per (referent, comparator, comp) triple, with rhm_*, phi_*, and
            optional sub_* summary columns.

        .csv:
            If save=True, will save the results dataframe to /results.

        .png:
            If plot=True, saves a tiled grid of npc heatmaps per metric.

        printed results:
            If display=True, prints per-component results in the terminal.
    """

    cl = [cluster] if cluster else []
    groups = df[group].unique()
    maindict = {g: df[df[group] == g].drop(labels=group, axis=1) for g in groups}
    pairings = list(combinations(groups, 2))

    scaler = StandardScaler()
    feat_cols = df.columns.drop([group, *cl])
    df_scaled = pd.DataFrame(scaler.fit_transform(df[feat_cols]), columns=feat_cols)
    _check_rank(df_scaled)

    # Global anchor: pooled-data PCA defines the canonical PC1..PC{npc} homologue. Every
    # per-fold PCA inside the bootstrap is Procrustes-aligned to this anchor (handled by
    # rhom when `anchor=` is set), so the bypc column index carries a consistent meaning
    # across all replicates and pairs.
    anchor_pca = basePCA(n_components=npc, rotation=rotation, method=method, corr=corr)
    anchor_pca.fit(df_scaled)
    anchor_loadings = anchor_pca.loadings.to_numpy()

    boot_model = rhom(rd=copy.deepcopy(df_scaled.values), bypc=True, n_comp=npc,
                      method=method, rotation=rotation, corr=corr,
                      anchor=anchor_loadings)
    cv = pair_cv(boot=True, k=folds, cluster=cluster)

    # engine.bypc stays False so cv.split is used (symmetric two-sided fold cross like
    # dir_proj). estimator.bypc=True still makes hom_pairs / pro_cong / subspace_sim
    # return per-component lists per replicate; we transpose at the bottom.
    boot_engine = BootstrapEngine(
        estimator=boot_model,
        cv=cv,
        pro_cong=True,
        shuffle=shuffle,
        subspace=subspace,
    )

    rows = []
    for ref, comp in pairings:
        print(f"Running By-Component Direct Projection: {ref} x {comp}")

        _check_rank(maindict[ref].drop(labels=cl, axis=1))
        _check_rank(maindict[comp].drop(labels=cl, axis=1))

        results = boot_engine(X=maindict[ref], y=maindict[comp], group=group)
        # results layout with estimator.bypc=True, engine.bypc=False:
        #   results[0]: list-of-lists [n_replicates × npc]  -- |r| per component per replicate
        #   results[1]: list-of-lists [n_replicates × npc]  -- TCC per component per replicate
        #   results[2]: list-of-lists [n_replicates × npc]  -- subspace cosines (if subspace=True)

        rhm_per_comp = list(map(list, zip(*results[0])))   # [npc × n_replicates]
        phi_per_comp = list(map(list, zip(*results[1])))

        # Subspace similarity is a whole-solution property -- collapse each replicate's
        # per-direction cosines to one mean and broadcast across this pair's npc rows.
        sub_per_pair = ([float(np.mean(s)) for s in results[2]] if subspace else None)

        for idx in range(npc):
            meta = {'referent': ref, 'comparator': comp, 'comp': idx + 1}
            row = _build_row(boot_model.n_comp, rhm_per_comp[idx], phi_per_comp[idx],
                             sub_data=sub_per_pair, metadata=meta)
            rows.append(row)

            if display:
                _display_stats(f"Direct Projection: {ref} x {comp} - Component {idx + 1}", row)

    dirproj_bypc_df = pd.DataFrame(rows)

    if plot:
        # Persist anchor loadings alongside the stats CSV and attach to the returned
        # frame so plot_dirproj_bypc can label panels with PC identity if desired.
        dirproj_bypc_df.attrs["loadings"] = anchor_pca.loadings.copy()

        from ..visualization.rhomplots import plot_dirproj_bypc, plot_aligned_wordclouds
        setupanalysis(path, file_prefix, includetime=False)
        plt.close('all')

        # Per-group full-data PCA -- one PCA per group on the group's entire data
        # (not a bootstrap fold). plot_aligned_wordclouds then Procrustes-aligns each
        # to the global anchor before rendering, so column k in every group's row
        # refers to the same homologue defined by the pooled reference.
        group_loadings = {
            g: basePCA(n_components=npc, rotation=rotation,
                       method=method, corr=corr).fit(
                maindict[g].drop(labels=cl, axis=1, errors='ignore')
            ).loadings
            for g in groups
        }
        wc_fig = plot_aligned_wordclouds(
            group_loadings,
            anchor_loadings=anchor_pca.loadings,
            n_features=len(df_scaled.columns),
            show_var=True,
            title="Per-group components (Procrustes-aligned to pooled reference)",
        )
        wc_fig.savefig(
            os.path.join(path, f"{file_prefix}/{file_prefix}_dj_bypc_wordclouds_{len(df_scaled.columns)}D_{npc}PC.png"),
            bbox_inches="tight", dpi=150,
        )
        plt.show()
        plt.close(wc_fig)

        metrics = ["rhm", "phi"] + (["sub"] if subspace else [])
        for name in metrics:
            fig = plot_dirproj_bypc(dirproj_bypc_df, metric=name)
            fig.savefig(
                os.path.join(path, f"{file_prefix}/{file_prefix}_dj_bypc_{len(df_scaled.columns)}D_{npc}PC_{name}.png"),
                bbox_inches="tight", dpi=150,
            )
            plt.show()
            plt.close(fig)

        loadings_path = os.path.join(
            path, f"{file_prefix}",
            f"{file_prefix}_loadings_{len(df_scaled.columns)}D_{npc}PC.csv",
        )
        anchor_pca.loadings.to_csv(loadings_path)

    if save:
        _export_report(dirproj_bypc_df, path, file_prefix,
                       f"dj_bypc_{len(df_scaled.columns)}D_{npc}PC")

    return dirproj_bypc_df

def omni_sample(df=None, group=None, npc=None, method='svd', rotation="varimax", corr='pearson',
                boot=1000, save=True, display=False, plot=True, shuffle=False, cluster=None,
                subspace=False, path='results', file_prefix=randint(10000, 99999)):
    """
    Omnibus-Sample Reproducibility
    ------------------------------
    This function conducts an omnibus-sample reproducibility analysis on your data.
    It randomly bootstrap reassigns halves of each level of an inputted grouping variable 
    to be used in either a 'sample' or 'omnibus' subset. The 'sample' subsets generate
    components representative of that level of the grouping variable, while the 'omnibus'
    subsets are aggregated with other groups to produce 'common' components. The analysis
    assesses the component similarity of the orthogonal aggregated set relative to each sample.
    It computes component similarity with:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)

    Parameters
    ----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        cluster: str, default=None
            Optional level-2 / clustering column (e.g. participant ID). When provided,
            whole clusters are kept together in every resample, fold, and split, so a unit
            never appears on both sides of a comparison. Prevents leakage and
            pseudoreplication with nested data. Requires at least `folds` distinct clusters
            per group where cross-validation is used.

        subspace: bool, default=False
            If True, also report subspace similarity via principal angles between the two
            loading subspaces (sub_* columns; rotation- and order-invariant, in [0, 1]).

        corr: str, default="pearson"
            Which correlation matrix to decompose under `method='eigen'`: "pearson",
            "spearman" (rank correlation, ordinal-friendly), or "polychoric" (latent
            correlation behind ordinal items via Olsson 1979 MLE; meaningful only for
            genuinely ordinal data and noticeably slower). Ignored under `method='svd'`,
            which is Pearson-only; pass `method='eigen'` to switch correlation type.

        npc: int, default=None
            Number of components to extract per solution.
        
        rotation: str, default="varimax"
            Rotation method to be performed on omnibus set. "none" for no rotation.

        boot: int, default=1000
            Number of bootstrap samples to generate 95% confidence intervals.
        
        save: bool, default=True
            Save outputted omnibus-sample reliability to .csv.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        path: str, default='results'
            The path to the output directory.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns
    -------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.
        
        .csv:
            If save=True, will save /results to a csv.

        printed results:
            If display=True, prints the output directly in the terminal.
    """

    drop_cols = [c for c in (group, cluster) if c is not None]
    samples = df[group].unique()
    df_t = df.drop(labels=drop_cols, axis=1)
    _check_rank(df_t)

    boot_model = rhom(rd=copy.deepcopy(df_t.values), n_comp=npc,
                      method=method, rotation=rotation, corr=corr)
    cv = pair_cv(omnibus=True, group=group, cluster=cluster, n=boot)
    
    # Initialize engine for omnibus resampling profile
    boot_engine = BootstrapEngine(
        estimator=boot_model,
        cv=cv,
        omnibus=True,
        pro_cong=True,
        shuffle=shuffle,
        subspace=subspace
    )

    rows = []
    total_rhm, total_phi, total_sub = [], [], []

    for sample in samples:
        print(f"Running Omnibus x {sample}")
        results = boot_engine(X=df, y=sample, group=group)

        total_rhm.extend(results[0])
        total_phi.extend(results[1])
        if subspace:
            total_sub.extend(results[2])

        row = _build_row(boot_model.n_comp, results[0], results[1],
                         sub_data=results[2] if subspace else None, metadata={group: sample})
        rows.append(row)

        if display:
            _display_stats(f'Omnibus x {sample}', row)

    # Append Global Summary Row
    total_row = _build_row(boot_model.n_comp, total_rhm, total_phi,
                           sub_data=total_sub if subspace else None, metadata={group: "Total"})
    rows.append(total_row)
    
    if display:
        _display_stats("Overall Omnibus Summary", total_row)

    omsamp_df = pd.DataFrame(rows)

    if plot:
        from ..visualization.rhomplots import plot_omni
        setupanalysis(path, file_prefix, includetime=False)

        plt.close('all')
        metrics = ["rhm", "phi"] + (["sub"] if subspace else [])
        for metric in metrics:
            fig = plot_omni(omsamp_df, metric=metric, title=f"Omnibus-Sample Reproducibility: {metric.upper()}", n_vars=len(df_t.columns))
            fig.savefig(os.path.join(path, f"{file_prefix}/{file_prefix}_omsamp_{len(df_t.columns)}D_{npc}PC_{metric}.png"), bbox_inches="tight", dpi=150)
            plt.show()
            plt.close(fig)

    if save:
        _export_report(omsamp_df, path, file_prefix, f"omsamp_{len(df_t.columns)}D_{npc}PC")
        
    return omsamp_df

def omni_variance(df=None, group=None, npc=None, method='svd', rotation='varimax',
                  corr='pearson', cluster=None, save=True, plot=True, display=False,
                  path='results', file_prefix=randint(10000, 99999)):
    """
    Omnibus Variance Attribution
    ----------------------------
    Fits one omnibus PCA on the pooled data and reports, for each group, the share
    of the group's standardised variance captured by each omnibus component. This
    answers "given these pooled-data components, how do they partition variance
    within each group?" -- a descriptive complement to the reproducibility builtins,
    which instead ask "do these components agree across groups?".

    The npc bars stacked together (Σₖ varₘ(k)) give the total variance captured by
    the omnibus solution in group g. See the Notes section for the formal definition
    and the random-subspace chance baseline.

    Parameters
    ----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        cluster: str, default=None
            Optional level-2 / clustering column. Stripped before decomposition so a
            cluster ID does not contaminate the PCA, but has no effect on the variance
            attribution itself (which is deterministic and does not resample).

        corr: str, default="pearson"
            Correlation matrix used for the omnibus fit under ``method='eigen'``:
            "pearson", "spearman", or "polychoric". Ignored under ``method='svd'``.

        npc: int, default=None
            Number of components to extract for the omnibus PCA.

        rotation: str, default="varimax"
            Rotation applied to the omnibus loadings. "none" for no rotation.

        save: bool, default=True
            Save the (group, comp, var_pct) table to .csv.

        plot: bool, default=True
            Render the stacked-bar variance attribution figure.

        display: bool, default=False
            Print each group's per-component variance share in the terminal.

        path: str, default='results'
            The path to the output directory.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files.

    Returns
    -------
        pd.DataFrame:
            One row per (group, comp) with columns ``n_comp``, ``group`` (named after
            your grouping variable), ``comp``, ``var_pct``.

        .csv:
            If save=True.

        .png:
            If plot=True, a stacked-bar figure with one bar per group and npc
            segments stacked by component contribution.

    Notes
    -----
    Let

    * $L \in \mathbb{R}^{p \times \text{npc}}$ represent the omnibus loadings (basePCA output)
    * $\ell_k = L[:, k] / \Vert{}L[:, k]\Vert{}$ represent the unit-normalised $k$-th column
    * $\tilde{X}_g \in \mathbb{R}^{n_g \times p}$ represent group $g$'s data after column-wise standardisation
    * $R_g = \tilde{X}_g^T \tilde{X}_g / n_g$ represent the within-group correlation matrix
    * $p$ represent the number of decomposed features

    Then component $k$'s variance share in group $g$ is

    $$\text{var}^g(k) = \frac{\ell_k^T \cdot R_g \cdot \ell_k}{p} \times 100\%$$

    which equals, equivalently, the projected-score formulation

    $$\text{var}^g(k) = \frac{\Vert{}\tilde{X}_g \cdot \ell_k\Vert{}^2}{n_g \cdot p} \times 100\%$$

    used inside the implementation. Summing over $k$ gives the total variance that
    the omnibus solution captures in group $g$ ($\le 100\%$; equal to $100\%$ only when
    $\text{npc} = p$).

    A random $\text{npc}$-dimensional subspace on standardised isotropic data captures, in
    expectation, $\frac{\text{npc}}{p} \times 100\%$ of the within-group variance, which the
    `plot_omni_variance` figure marks with a dashed horizontal line. Because
    variance is a *quadratic* projection, this baseline is the *square* of the
    canonical-correlation baseline $\sqrt{\frac{\text{npc}}{p}}$ used for the
    reproducibility metrics (rhm, phi, sub).

    .. math::

        \\mathrm{var}_g(k) = \\frac{\\ell_k^\\top R_g \\,\\ell_k}{p} \\times 100\\%,
        \\qquad
        \\text{chance} \\approx \\frac{\\mathrm{npc}}{p} \\times 100\\%.
    """

    drop_cols = [c for c in (group, cluster) if c is not None]
    feat_cols = df.columns.drop(drop_cols)
    _check_rank(df[feat_cols])

    # Omnibus PCA on pooled data defines the reference component set.
    omni = basePCA(n_components=npc, rotation=rotation, method=method, corr=corr)
    omni.fit(df[feat_cols])
    L = omni.loadings.values

    # basePCA stores loadings as eigvec * sqrt(eigval); for variance attribution we
    # want unit-norm directions, so divide each column by its norm.
    norms = np.linalg.norm(L, axis=0)
    L_unit = L / norms

    p = len(feat_cols)
    samples = df[group].unique()

    rows = []
    for g in samples:
        X_g = df.loc[df[group] == g, feat_cols]
        X_g_std = StandardScaler().fit_transform(X_g)
        scores = X_g_std @ L_unit                                # (n_g, npc)
        var_per_comp = (scores ** 2).sum(axis=0) / len(X_g_std)  # variance of each PC's scores
        var_pct = var_per_comp / p * 100                         # share of group's total variance

        for k in range(npc):
            rows.append({
                "n_comp": f"{npc}PC",
                group: g,
                "comp": k + 1,
                "var_pct": float(var_pct[k]),
            })

        if display:
            shares = ", ".join(f"PC{k+1}={var_pct[k]:.1f}%" for k in range(npc))
            total = float(var_pct.sum())
            print(f"{g}: total={total:.1f}% | {shares}")

    omni_var_df = pd.DataFrame(rows)

    if plot:
        from ..visualization.rhomplots import plot_omni_variance
        setupanalysis(path, file_prefix, includetime=False)
        plt.close('all')

        fig = plot_omni_variance(omni_var_df, group=group, n_vars=p)
        fig.savefig(
            os.path.join(path, f"{file_prefix}/{file_prefix}_omni_var_{p}D_{npc}PC.png"),
            bbox_inches="tight", dpi=150,
        )
        plt.show()
        plt.close(fig)

    if save:
        _export_report(omni_var_df, path, file_prefix, f"omni_var_{p}D_{npc}PC")

    return omni_var_df

def omsamp_bypc(df=None, group=None, npc=None, method='svd', rotation="varimax", corr='pearson',
         folds=5, save=True, plot=True, display=False, shuffle=False, cluster=None,
         subspace=False, path='results', file_prefix=randint(10000, 99999)):
    
    """    
    Omnibus-Sample Reproducibility: By-Component
    ------------------------------
    This function conducts an omnibus-sample reproducibility analysis on your data,
    modified to assess the correspondence between each component of an omnibus solution and its
    corresponding components in each subset. It randomly reassigns halves of each level 
    of an inputted grouping variable to be stably used in either a 'sample' or 'omnibus' subset. 
    The 'sample' subsets are folded to generate cross-validated components representative of that level of
    the grouping variable, while the 'omnibus' subset is aggregated with other groups to produce 'common' components.
    The analysis assesses the component similarity of the orthogonal aggregated set relative to each sample.
    It computes component similarity with:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)

    Parameters
    ----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        cluster: str, default=None
            Optional level-2 / clustering column (e.g. participant ID). When provided,
            whole clusters are kept together in every resample, fold, and split, so a unit
            never appears on both sides of a comparison. Prevents leakage and
            pseudoreplication with nested data. Requires at least `folds` distinct clusters
            per group where cross-validation is used.

        subspace: bool, default=False
            If True, also report subspace similarity via principal angles between the two
            loading subspaces (sub_* columns; rotation- and order-invariant, in [0, 1]).

        corr: str, default="pearson"
            Which correlation matrix to decompose under `method='eigen'`: "pearson",
            "spearman" (rank correlation, ordinal-friendly), or "polychoric" (latent
            correlation behind ordinal items via Olsson 1979 MLE; meaningful only for
            genuinely ordinal data and noticeably slower). Ignored under `method='svd'`,
            which is Pearson-only; pass `method='eigen'` to switch correlation type.

        npc: int, default=None
            Number of components to extract per solution.
        
        rotation: str, default="varimax"
            Rotation method to be performed on omnibus set. "none" for no rotation.

        folds: int, default=5
            Number of folds to use for cross-validation.
        
        save: bool, default=True
            Save outputted omnibus-sample reliability to .csv.

        plot: bool, default=True
            Save wordclouds, a Scree plot, and .csv files for the specified omnibus set.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        path: str, default='results'
            The path to the output directory. 

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns
    -------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.
        
        .csv:
            If save=True, will save /results to a csv.

        RHom PCA results:
            If plot=True, will save the results, including wordclouds and Scree plot, for the omnibus set in a RHom folder.

        printed results:
            If display=True, prints the output directly in the terminal.
    """

    drop_cols = [c for c in (group, cluster) if c is not None]
    df_t = df.drop(labels=drop_cols, axis=1)
    _check_rank(df_t)
    boot_model = rhom(rd=copy.deepcopy(df_t.values), bypc=True, n_comp=npc,
                      method=method, rotation=rotation, corr=corr)
    cv = pair_cv(boot=True, group=group, cluster=cluster, k=folds)
    
    nval = (df[group].value_counts().min()) / 2
    maindict = cv.omni_prep(df=df, subrows=nval)
    samples = df[group].unique()
    
    # Initialize engine tailored for granular by-component splits
    boot_engine = BootstrapEngine(
        estimator=boot_model,
        cv=cv,
        bypc=True,
        pro_cong=True,
        shuffle=shuffle,
        subspace=subspace
    )

    rows = []
    for sample in samples:
        print(f"Running Component Breakdown: omnibus x {sample}")
        # Returns [complist, philist(, sublist)] -> [n_components, n_fold_combinations].
        # sublist (if present) is sample-level subspace similarity per fold combination, not
        # per-component, so it is reused across this sample's component rows.
        comps = boot_engine(X=maindict['omnibus'], y=maindict[sample], group=group)
        sub = comps[2] if subspace else None

        for idx in range(npc):
            meta = {group: sample, 'comp': idx + 1}
            row = _build_row(boot_model.n_comp, comps[0][idx], comps[1][idx],
                             sub_data=sub, metadata=meta)
            rows.append(row)
            
            if display:
                _display_stats(f"Omnibus x {sample} - Component {idx + 1}", row)

    stats_bypc = pd.DataFrame(rows)
    
    if plot:
        # Fit basePCA on the exact omnibus half drawn in this call, so the wordclouds
        # match the loadings that underlie the per-component similarity scores above.
        om_model = basePCA(n_components=npc, rotation=rotation, method=method)
        om_model.fit(maindict['omnibus'])

        cloud_dir = setupanalysis(os.path.join(path, file_prefix), "bypc_wordclouds", includetime=False)
        save_wordclouds(om_model.loadings, path=str(cloud_dir))

        # Persist the loadings alongside the stats CSV and attach them to the returned
        # frame so plot_bypc can render the wordclouds without a save/reload round-trip.
        loadings_path = os.path.join(
            path, f"{file_prefix}",
            f"{file_prefix}_loadings_{len(df_t.columns)}D_{npc}PC.csv",
        )
        om_model.loadings.to_csv(loadings_path)
        stats_bypc.attrs["loadings"] = om_model.loadings.copy()

        from ..visualization.rhomplots import plot_bypc
        plt.close('all')
        metrics = ["rhm", "phi"] + (["sub"] if subspace else [])
        for metric in metrics:
            fig = plot_bypc(stats_bypc, metric=metric)
            fig.savefig(os.path.join(path, f"{file_prefix}/{file_prefix}_bypc_{len(df_t.columns)}D_{npc}PC_{metric}.png"), bbox_inches="tight", dpi=150)
            plt.show()
            plt.close(fig)

    if save:
        _export_report(stats_bypc, path, file_prefix, f"bypc_{len(df_t.columns)}D_{npc}PC")

    return stats_bypc

def holdout_cv(df=None, group=None, folds=None, npc=None, method='svd', rotation='varimax',
               corr='pearson', cluster=None, save=True, plot=True, display=False,
               shuffle=False, subspace=False, path='results',
               file_prefix=randint(10000, 99999)):
    """
    Held-Out Cross-Validation
    -------------------------
    Cross-validated component reproducibility: in each fold the PCA is fit on the
    training rows and the held-out rows independently, and the two solutions are
    compared via rhm / phi (and optionally sub). Three modes, selected by which of
    ``group`` / ``folds`` are passed:

        * ``group='colname'`` only -- leave-one-group-out: each level of ``colname``
          is held out as the test set once.
        * ``folds=K`` only -- random K-fold partition (cluster-aware when
          ``cluster=`` is set, so whole clusters land in one fold).
        * ``group='colname'`` AND ``folds=K`` -- stratified K-fold: each fold pulls
          proportional rows from every level of ``colname`` (cluster-aware within
          each stratum when ``cluster=`` is also set).

    Each fold returns point estimates (no bootstrap inside a fold); the final
    "summary" row holds the mean and across-fold CI of each metric.

    Parameters
    ----------

        df: pd.Dataframe, default=None
            Decomposition columns plus the grouping / clustering columns if used.

        group: str, default=None
            Column name driving leave-one-group-out splits when used alone, or the
            stratifier when paired with ``folds=K``.

        folds: int, default=None
            Number of K-fold splits. Random partition when used alone; stratified by
            ``group`` when both are passed.

        cluster: str, default=None
            Optional level-2 / clustering column. With ``folds=K`` (random or
            stratified), whole clusters are kept together within a fold (requires at
            least K distinct clusters; in stratified mode, K within each stratum).
            With ``group=...`` only, cluster is irrelevant since groups already define
            the partition.

        subspace: bool, default=False
            If True, also report subspace similarity via principal angles.

        corr: str, default="pearson"
            Correlation matrix for ``method='eigen'``: "pearson", "spearman", or
            "polychoric". Ignored under ``method='svd'``.

        npc: int, default=None
            Number of components to extract per fold.

        rotation: str, default="varimax"
            Rotation applied to each fold's PCA.

        shuffle: bool, default=False
            If True, Mantel-shuffle the feature columns first (destroys cross-variable
            structure while preserving marginals) to produce a noise-floor null run.

        save / plot / display / path / file_prefix:
            Same conventions as the other builtins.

    Returns
    -------
        pd.DataFrame:
            One row per fold with ``rhm_x`` / ``phi_x`` (and ``sub_x`` if requested)
            point estimates, plus a final ``fold='summary'`` row holding the mean
            and across-fold CI for each metric.

        .csv:
            If save=True.

        .png:
            If plot=True, one horizontal-bar figure per metric via ``plot_omni``.
    """
    if group is None and folds is None:
        raise ValueError(
            "holdout_cv requires at least one of group= (leave-one-group-out) or "
            "folds= (k-fold). Got both None."
        )

    drop_cols = [c for c in (group, cluster) if c is not None]
    feat_cols = df.columns.drop(drop_cols) if drop_cols else df.columns
    _check_rank(df[feat_cols])

    # Optional Mantel shuffle for the null baseline: column-wise independent
    # permutation destroys cross-variable structure while preserving marginals.
    # Pass only the feature columns so fullmantel doesn't accidentally treat a
    # numeric cluster ID as a feature; reassign by .values to overwrite in place.
    df_input = df.copy()
    if shuffle:
        from ..preprocessing.data_utils import fullmantel
        df_input[feat_cols] = fullmantel(df_input[feat_cols]).values

    # Mode dispatch: pair_cv's stratified_kfold flag flips holdout_split into the
    # stratified-K-fold path; otherwise self.group toggles LOGO and falling through
    # gives plain K-fold.
    if group is not None and folds is not None:
        cv = pair_cv(k=folds, cluster=cluster, stratify=group, stratified_kfold=True)
        mode = f"{folds}-fold stratified by '{group}'"
    elif group is not None:
        cv = pair_cv(group=group, cluster=cluster)
        mode = f"leave-one-group-out by '{group}'"
    else:
        cv = pair_cv(k=folds, cluster=cluster)
        mode = f"{folds}-fold"

    boot_model = rhom(rd=copy.deepcopy(df_input[feat_cols].values), n_comp=npc,
                      method=method, rotation=rotation, corr=corr)

    rows = []
    rhm_vals, phi_vals, sub_vals = [], [], []

    print(f"Running Held-Out CV ({mode})")
    for train, test, label in cv.holdout_split(df_input):
        boot_model.fit(train, test)
        preds = boot_model.predict()
        corrs = np.corrcoef(preds[0], preds[1], rowvar=False)

        rhm_val = float(boot_model.hom_pairs(corrs))
        phi_val = float(boot_model.pro_cong())
        sub_val = float(boot_model.subspace_sim()) if subspace else None

        rhm_vals.append(rhm_val)
        phi_vals.append(phi_val)
        if subspace:
            sub_vals.append(sub_val)

        # Per-fold row: pass a single-value "distribution" to _build_row so the schema
        # matches the summary row exactly (CIs and SE collapse to the point estimate;
        # zero-width error bars on the plot mark these as deterministic per-fold scores).
        meta = {"fold": label}
        row = _build_row(boot_model.n_comp, [rhm_val], [phi_val],
                         sub_data=[sub_val] if subspace else None, metadata=meta)
        rows.append(row)

        if display:
            extras = f", sub={sub_val:.3g}" if subspace else ""
            print(f"  held out '{label}': rhm={rhm_val:.3g}, phi={phi_val:.3g}{extras}")

    # Cross-fold summary: same _build_row call but fed the actual fold-level distribution,
    # so rhm_se / rhm_LCI / rhm_UCI report the across-fold spread.
    summary_row = _build_row(
        boot_model.n_comp, rhm_vals, phi_vals,
        sub_data=sub_vals if subspace else None,
        metadata={"fold": "summary"},
    )
    rows.append(summary_row)

    if display:
        _display_stats(f"Held-Out CV summary ({mode})", summary_row)

    holdout_df = pd.DataFrame(rows)

    if plot:
        from ..visualization.rhomplots import plot_omni
        setupanalysis(path, file_prefix, includetime=False)
        plt.close('all')

        metrics = ["rhm", "phi"] + (["sub"] if subspace else [])
        for name in metrics:
            # plot_omni's "Total" guard drops the omni_sample summary row; here we want
            # the summary row visible (it's the headline number), so we pass group='fold'
            # explicitly and rely on the label being 'summary' (not 'Total').
            fig = plot_omni(holdout_df, group="fold", metric=name,
                            n_vars=len(feat_cols),
                            title=f"Held-Out CV ({mode}): {name.upper()}")
            fig.savefig(
                os.path.join(path, f"{file_prefix}/{file_prefix}_holdout_cv_{len(feat_cols)}D_{npc}PC_{name}.png"),
                bbox_inches="tight", dpi=150,
            )
            plt.show()
            plt.close(fig)

    if save:
        _export_report(holdout_df, path, file_prefix,
                       f"holdout_cv_{len(feat_cols)}D_{npc}PC")

    return holdout_df


