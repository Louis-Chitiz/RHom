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
              boot=1000, save=True, display=False, shuffle=False, cluster=None, subspace=False,
              path='results', file_prefix=randint(10000, 99999)):
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
    
    drop_cols = [c for c in (group, cluster) if c is not None]
    df_t = df.drop(labels=drop_cols, axis=1) if drop_cols else df
    samples = df[group].unique() if group else ['fulldata']
    _check_rank(df_t)

    boot_model = rhom(rd=copy.deepcopy(df_t.values), n_comp=npc,
                      method=method, rotation=rotation, corr=corr)
    cv = pair_cv(group=group, cluster=cluster, n=boot)
    
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

def bypc(df=None, group=None, npc=None, method='svd', rotation="varimax", corr='pearson',
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