from .._deps import pd, randint, plt

import os
from datetime import datetime
from pathlib import Path

from ..visualization.pcastats import plot_scree, plot_stats, display_explained_variance
from ..visualization.wordclouds import save_wordclouds

def setupanalysis(
    path: str = None, pathprefix: str = "analysis", includetime: bool = True
) -> Path:
    """
    Setup a folder for analysis.

    Args:
        path (str): Path to the folder where the analysis folder should be created.
        pathprefix (str): Prefix for the analysis folder.
        includetime (bool): If True, the analysis folder will be named with a timestamp.

    Returns:
        str: Path to the analysis folder.
    """
    # Use current working directory if no path is provided
    base_path = Path(path) if path else Path.cwd()
    
    # Construct folder name
    folder_name = pathprefix
    if includetime:
        timestamp = datetime.now().strftime("%d%m%Y")
        folder_name = f"{pathprefix}_{timestamp}"
    
    full_path = base_path / folder_name
    
    if full_path.exists():
        print(f"Warning: {full_path} already exists. Results may be overwritten.")
        
    full_path.mkdir(parents=True, exist_ok=True)

    return full_path

def save_pca_results(folder_path: Path, data_map: dict):
    """Helper to write CSVs to a specific path.""" 
    for filename, df in data_map.items():
        if df is not None:
            df.to_csv(folder_path / f"{filename}.csv")

def save_pca_figures(folder_path: Path, data_map: dict):
    """Helper to write CSVs to a specific path.""" 
    for filename, figure in data_map.items():
        if figure is not None:
            figure.savefig(folder_path / f"{filename}.png", bbox_inches='tight')

def run_save_sequence(pca, path=None, pathprefix=None):
    """
    Orchestrates saving results into a structured directory: 
    /csvdata, /wordclouds, and /statistics.
    """
    # Directory Initialization
    if not pca.path:
        pca.path = setupanalysis(path, pathprefix)
    
    root = Path(pca.path)
    
    # Define subfolders and ensure they exist
    subdirs = {
        "csv": root / "csvdata",
        "clouds": root / "wordclouds",
        "stats": root / "statistics"
    }
    
    for folder in subdirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    # CSV Data
    # Using .get() or getattr() allows saving even if transform() wasn't called
    main_manifest = {
        "loadings": getattr(pca, "loadings", None),
        "eigenvalues": pd.DataFrame(getattr(pca, "eigenvalues", []), columns=["variance"]),
        "variance_stats": display_explained_variance(pca, verbosity = 0) if hasattr(pca, "eigenvalues") else None,
        "fitted_results": getattr(pca, "extra_columns", None),
        "projected_results": getattr(pca, "project_columns", None),
        "original_data": getattr(pca, "ogdf", None)
    }

    # Filter out None values so save_pca_results doesn't try to save empty files
    save_pca_results(subdirs["csv"], {k: v for k, v in main_manifest.items() if v is not None})

    # Figures
    # Mapping logic to figure functions
    desc_manifest = {}

    # Scree plots
    desc_manifest["scree_eigens"] = plot_scree(pca, type="eigenvals")
    desc_manifest["scree_expvar"] = plot_scree(pca)
    
    if hasattr(pca, "_raw_fitted"):
        desc_manifest["fitted_means"] = plot_stats(pca._raw_fitted)
    if hasattr(pca, "_raw_project"):
        desc_manifest["projected_means"] = plot_stats(pca._raw_project)
    
    save_pca_figures(subdirs["stats"], desc_manifest)
    plt.close('all') # Comprehensive cleanup
    
    # Wordclouds (Internal path handling)
    save_wordclouds(pca.loadings, path=subdirs["clouds"])



        