from RHom._deps import os, pd, np, copy, randint, norm, plt, sns, combinations, StandardScaler

from RHom.core.base_pca import basePCA
from RHom.rhom import bootstrap
from RHom.rhom.metrics import rhom
from RHom.rhom.resampling import pair_cv
from RHom.rhom.bootstrap import BootstrapEngine
from RHom.io.save import setupanalysis

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

def _build_row(n_comp, rhm_data, phi_data=None, metadata=None):
    """Assembles a single unified row mapping for reporting metrics."""
    row = {"n_comp": f"{n_comp}PC" if isinstance(n_comp, int) else n_comp}
    if metadata:
        row.update(metadata)
        
    rhm_stats = _summary_stats(rhm_data)
    row.update({f"rhm_{k}": v for k, v in rhm_stats.items()})
    
    if phi_data is not None:
        phi_stats = _summary_stats(phi_data)
        row.update({f"phi_{k}": v for k, v in phi_stats.items()})
        
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

def splithalf(df=None, group=None, npc=None, method='svd', rotation='varimax', 
              boot=1000, save=True, display=False, shuffle=False, 
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
    
    df_t = df.drop(labels=[group], axis=1) if group else df
    samples = df[group].unique() if group else ['fulldata']
    
    boot_model = rhom(rd=copy.deepcopy(df_t.values), n_comp=npc, method=method, rotation=rotation)
    cv = pair_cv(group=group, n=boot)
    
    boot_engine = BootstrapEngine(
        estimator=boot_model, 
        cv=cv, 
        splithalf=True, 
        pro_cong=True, 
        shuffle=shuffle
    )
    
    rows = []
    for sample in samples:
        print(f"Running Split-Half: {sample}")
        # Execute via the unified __call__ interface
        results = boot_engine(X=df, y=sample, group=group)
        
        meta = {group: sample} if group else {"Group": "fulldata"}
        row = _build_row(boot_model.n_comp, results[0], results[1], metadata=meta)
        rows.append(row)
        
        if display:
            _display_stats(f"Split-Half Reliability for {sample}", row)

    split_df = pd.DataFrame(rows)
    if save:
        _export_report(split_df, path, file_prefix, f"splithalf_{len(df_t.columns)}D_{npc}PC")
        
    return split_df

def dir_proj(df=None, group=None, npc=None, method='svd', rotation="varimax", folds=5,
             save=True, plot=True, display=False, shuffle=False, 
             path='results', file_prefix=randint(10000, 99999)):
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
    groups = df[group].unique()
    maindict = {g: df[df[group] == g].drop(labels=group, axis=1) for g in groups}
    pairings = list(combinations(groups, 2))
    
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(labels=group, axis=1)), columns=df.columns.drop(group))
    
    boot_model = rhom(rd=copy.deepcopy(df_scaled.values), n_comp=npc, method=method, rotation=rotation)
    cv = pair_cv(boot=True, k=folds)
    
    boot_engine = BootstrapEngine(
        estimator=boot_model, 
        cv=cv, 
        pro_cong=True, 
        shuffle=shuffle
    )
    
    rows = []
    dirproj_mtx = pd.DataFrame(1.0, columns=groups, index=groups) if plot else None
    dirproj_phi = pd.DataFrame(1.0, columns=groups, index=groups) if plot else None
    
    for ref, comp in pairings:
        print(f"Running Matrix Path: {ref} x {comp}")
        # Execute using referent and comparator subsets
        results = boot_engine(X=maindict[ref], y=maindict[comp], group=group)
        
        meta = {'referent': ref, 'comparator': comp}
        row = _build_row(boot_model.n_comp, results[0], results[1], metadata=meta)
        rows.append(row)
        
        if plot:
            for mtx, metric in zip([dirproj_mtx, dirproj_phi], ['rhm_x', 'phi_x']):
                mtx.loc[ref, comp] = mtx.loc[comp, ref] = row[metric]
                
        if display:
            _display_stats(f"Direct Projection: {ref} x {comp}", row)

    dirproj_df = pd.DataFrame(rows)
    
    if plot:
        plt.close('all')
        for mtx, name, label in [(dirproj_mtx, 'rhm', 'Mean Homologue Similarity'), (dirproj_phi, 'phi', 'Mean Factor Congruence')]:
            sns.heatmap(mtx, vmin=mtx.values.min(), annot=True, annot_kws={"fontsize": 35 / np.sqrt(len(mtx))}, cmap="flare")
            plt.suptitle(label, fontsize=16)
            plt.savefig(os.path.join(path, f"{file_prefix}/{file_prefix}_heatmap{len(df_scaled.columns)}D_{npc}PC_{name}.png"))
            plt.show(); plt.close()
            
    if save:
        _export_report(dirproj_df, path, file_prefix, f"dj{len(df_scaled.columns)}D_{npc}PC")
        
    return dirproj_df

def omni_sample(df=None, group=None, npc=None, method='svd', rotation="varimax", boot=1000,
                save=True, display=False, shuffle=False, path='results', file_prefix=randint(10000, 99999)):
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

    samples = df[group].unique()
    df_t = df.drop(labels=[group], axis=1)
    
    boot_model = rhom(rd=copy.deepcopy(df_t.values), n_comp=npc, method=method, rotation=rotation)
    cv = pair_cv(omnibus=True, group=group, n=boot)
    
    # Initialize engine for omnibus resampling profile
    boot_engine = BootstrapEngine(
        estimator=boot_model, 
        cv=cv, 
        omnibus=True, 
        pro_cong=True, 
        shuffle=shuffle
    )
    
    rows = []
    total_rhm, total_phi = [], []
    
    for sample in samples:
        print(f"Running Omnibus x {sample}")
        results = boot_engine(X=df, y=sample, group=group)
        
        total_rhm.extend(results[0])
        total_phi.extend(results[1])
        
        row = _build_row(boot_model.n_comp, results[0], results[1], metadata={group: sample})
        rows.append(row)
        
        if display:
            _display_stats(f'Omnibus x {sample}', row)

    # Append Global Summary Row
    total_row = _build_row(boot_model.n_comp, total_rhm, total_phi, metadata={group: "Total"})
    rows.append(total_row)
    
    if display:
        _display_stats("Overall Omnibus Summary", total_row)

    omsamp_df = pd.DataFrame(rows)

    if save:
        _export_report(omsamp_df, path, file_prefix, f"omsamp_{len(df_t.columns)}D_{npc}PC")
        
    return omsamp_df

def bypc(df=None, group=None, npc=None, method='svd', rotation="varimax", folds=5,
         save=True, plot=True, display=False, shuffle=False, path='results', file_prefix=randint(10000, 99999)):
    
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

    df_t = df.drop(labels=group, axis=1)
    boot_model = rhom(rd=copy.deepcopy(df_t.values), bypc=True, n_comp=npc, method=method, rotation=rotation)
    cv = pair_cv(boot=True, group=group, k=folds)
    
    nval = (df[group].value_counts().min()) / 2
    maindict = cv.omni_prep(df=df, subrows=nval)
    samples = df[group].unique()
    
    # Initialize engine tailored for granular by-component splits
    boot_engine = BootstrapEngine(
        estimator=boot_model, 
        cv=cv, 
        bypc=True, 
        pro_cong=True, 
        shuffle=shuffle
    )
    
    rows = []
    for sample in samples:
        print(f"Running Component Breakdown: omnibus x {sample}")
        # Returns shape: [complist, philist] -> [n_components, n_folds_combinations]
        comps = boot_engine(X=maindict['omnibus'], y=maindict[sample], group=group)
        
        for idx in range(npc):
            meta = {group: sample, 'comp': idx + 1}
            row = _build_row(boot_model.n_comp, comps[0][idx], comps[1][idx], metadata=meta)
            rows.append(row)
            
            if display:
                _display_stats(f"Omnibus x {sample} - Component {idx + 1}", row)

    stats_bypc = pd.DataFrame(rows)
    
    if plot:
        model = basePCA(n_components=npc, rotation=rotation)
        model.fit_transform(maindict['omnibus'])
        model.save(path=os.path.join(path, f"{file_prefix}"), pathprefix=f"{file_prefix}_os")
        
    if save:
        _export_report(stats_bypc, path, file_prefix, f"bypc_{len(df_t.columns)}D_{npc}PC")
        
    return stats_bypc