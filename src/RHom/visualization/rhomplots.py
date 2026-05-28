from .._deps import pd, np, plt

import seaborn as sns


def plot_omni(results: pd.DataFrame, group: str = None, metric: str = "rhm",
              title: str = None, n_vars: int = None):
    """
    Horizontal bar plot of omnibus-sample reproducibility with 95% CI error bars.

    Parameters
    ----------
        results: pd.DataFrame
            The output of omni_sample (one row per group level plus a "Total" row).
        group: str, default=None
            Name of the grouping column to label the bars. If None, it is inferred as the
            only column that is not `n_comp` or a metric column.
        metric: str, default="rhm"
            Which similarity metric to plot: "rhm" (homologue similarity), "phi"
            (factor congruence), or "sub" (subspace similarity via principal angles).
        title: str, default=None
            Plot title. Defaults to the metric's full name.
        n_vars: int, default=None
            Number of variables that were decomposed (p). Only used when metric="sub":
            if provided, a dashed vertical line is drawn at the chance baseline
            sqrt(npc / n_vars), since subspace similarity has a non-zero null that
            depends on npc/p.

    Returns
    -------
        matplotlib.figure.Figure
    """
    metric = metric.lower()
    labels_for = {
        "rhm": "Mean Homologue Similarity",
        "phi": "Mean Factor Congruence",
        "sub": "Mean Subspace Similarity",
    }
    if metric not in labels_for:
        raise ValueError(f"metric must be one of {list(labels_for)}, got '{metric}'.")
    if f"{metric}_x" not in results.columns:
        raise ValueError(
            f"Column '{metric}_x' not found in results. "
            f"Was omni_sample run with subspace=True?" if metric == "sub"
            else f"Column '{metric}_x' not found in results. Was omni_sample run with this metric?"
        )

    # Infer the grouping column if not supplied
    if group is None:
        reserved = {"n_comp"}
        group = next(c for c in results.columns
                     if c not in reserved and not c.startswith(("rhm_", "phi_", "sub_")))
        
    results = results[results[group] != "Total"]

    labels = results[group].astype(str).tolist()
    x = results[f"{metric}_x"].to_numpy()
    lci = results[f"{metric}_LCI"].to_numpy()
    uci = results[f"{metric}_UCI"].to_numpy()

    # Asymmetric error bars from the percentile CIs; clip tiny negatives from rounding
    xerr = np.clip(np.vstack([x - lci, uci - x]), 0, None)

    pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(labels) + 1.5))

    ax.barh(pos, x, xerr=xerr, capsize=4, color="#4C72B0", ecolor="0.3")
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # first row at the top
    ax.set_xlim(0, 1)
    ax.set_xlabel(labels_for[metric])
    ax.set_title(title if title is not None else labels_for[metric])

    # Value in the centre of each bar, leading zero dropped (.83 not 0.83)
    for i, val in zip(pos, x):
        ax.text(val / 2, i, f"{val:.2f}".replace("-0.", "-.").lstrip("0") or "0",
                ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    # Chance baseline for subspace similarity (depends on npc / n_vars; not zero)
    if metric == "sub" and n_vars:
        npc = int(str(results["n_comp"].iloc[0]).rstrip("PC"))
        chance = float(np.sqrt(npc / n_vars))
        ax.axvline(chance, linestyle="--", color="0.3", linewidth=1.2, zorder=3)
        ax.text(chance + 0.01, 0.99,
                f"chance ≈ {chance:.2f}".replace("0.", "."),
                transform=ax.get_xaxis_transform(),
                color="0.3", fontsize=9, va="top")

    fig.tight_layout()
    return fig


def plot_dirproj(results: pd.DataFrame, metric: str = "rhm", title: str = None):
    """
    Lower-triangle heatmap of direct-projection results for one similarity metric.

    Reconstructs the symmetric pairwise matrix from the `referent`/`comparator` rows of
    `dir_proj`'s output and plots it as a heatmap, masking the diagonal and the upper
    triangle so each group pair appears once.

    Parameters
    ----------
        results: pd.DataFrame
            Output of dir_proj (one row per group pair with 'referent', 'comparator', and
            per-metric '*_x' columns).
        metric: str, default="rhm"
            Which similarity metric to plot: "rhm" (homologue similarity), "phi"
            (factor congruence), or "sub" (subspace similarity).
        title: str, default=None
            Plot title. Defaults to the metric's full name.

    Returns
    -------
        matplotlib.figure.Figure
    """
    metric = metric.lower()
    labels_for = {
        "rhm": "Mean Homologue Similarity",
        "phi": "Mean Factor Congruence",
        "sub": "Mean Subspace Similarity",
    }
    if metric not in labels_for:
        raise ValueError(f"metric must be one of {list(labels_for)}, got '{metric}'.")
    col = f"{metric}_x"
    if col not in results.columns:
        raise ValueError(
            f"Column '{col}' not found in results. "
            f"Was dir_proj run with subspace=True?" if metric == "sub"
            else f"Column '{col}' not found in results. Was dir_proj run with this metric?"
        )

    # Reconstruct the symmetric pairwise matrix from the row records
    groups = pd.Index(pd.unique(pd.concat([results["referent"], results["comparator"]])))
    mtx = pd.DataFrame(1.0, index=groups, columns=groups)
    for _, row in results.iterrows():
        ref, comp = row["referent"], row["comparator"]
        mtx.loc[ref, comp] = mtx.loc[comp, ref] = row[col]

    n = len(mtx)
    # Show each pair once: hide the diagonal and the redundant upper mirror
    mask = np.triu(np.ones((n, n), dtype=bool))
    lower = ~mask
    cell = max(0.5, min(1.0, 10.0 / n))   # inches per cell; caps the figure as n grows
    fs = max(6, cell * 13)                # annotation + label font tracks cell size

    shown = mtx.values[lower]
    annot = np.vectorize(lambda v: f"{v:.2f}".lstrip("0"))(mtx.values)
    label = labels_for[metric]

    fig, ax = plt.subplots(figsize=(n * cell + 2, n * cell + 2))
    sns.heatmap(
        mtx,
        mask=mask,
        vmin=shown.min(),
        vmax=shown.max(),
        annot=annot,
        fmt="",
        annot_kws={"fontsize": fs},
        cmap="flare",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.6, "label": label},
        ax=ax,
    )
    ax.set_title(title if title is not None else label, fontsize=16, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fs)
    fig.tight_layout()
    return fig


def plot_bypc(stats: pd.DataFrame, loadings: pd.DataFrame = None,
              group: str = None, metric: str = "rhm",
              title: str = None, font: str = "helvetica"):
    """
    Per-component panel plot: omnibus wordcloud next to a horizontal bar of that
    component's similarity to each group.

    Parameters
    ----------
        stats: pd.DataFrame
            Output of bypc (one row per group x component with rhm_x / phi_x / sub_x
            columns and a 'comp' column indicating the component index).
        loadings: pd.DataFrame, default=None
            Omnibus loadings (rows = items, columns = PC1..PCk). If None, the function
            falls back to stats.attrs["loadings"], which bypc attaches automatically when
            plot=True. Pass it explicitly if you loaded `stats` from disk.
        group: str, default=None
            Name of the grouping column in `stats`. Inferred from the column layout if not
            supplied.
        metric: str, default="rhm"
            Which similarity metric to display in the bar panels: "rhm", "phi", or "sub".
        title: str, default=None
            Figure-level title. Defaults to the metric's full name.
        font: str, default="helvetica"
            Font used for the wordclouds (must be present in RHom/visualization/fonts).

    Returns
    -------
        matplotlib.figure.Figure
    """
    import os as _os
    from wordcloud import WordCloud
    from .wordclouds import create_dynamic_mask

    if loadings is None:
        loadings = stats.attrs.get("loadings")
        if loadings is None:
            raise ValueError(
                "Omnibus loadings not provided and not found in stats.attrs['loadings']. "
                "Pass them explicitly via the `loadings` argument."
            )

    # Normalise: if `loadings` was read from CSV without index_col=0, the item names sit in
    # a non-numeric column rather than on the index. Promote the first such column.
    non_numeric = loadings.select_dtypes(exclude="number").columns
    if len(non_numeric) >= 1:
        loadings = loadings.set_index(non_numeric[0])

    metric = metric.lower()
    labels_for = {
        "rhm": "Mean Homologue Similarity",
        "phi": "Mean Factor Congruence",
        "sub": "Mean Subspace Similarity",
    }
    if metric not in labels_for:
        raise ValueError(f"metric must be one of {list(labels_for)}, got '{metric}'.")
    col = f"{metric}_x"
    if col not in stats.columns:
        raise ValueError(f"Column '{col}' not found in stats.")

    # Infer the grouping column from the stats layout if not provided
    if group is None:
        reserved = {"n_comp", "comp"}
        group = next(c for c in stats.columns
                     if c not in reserved and not c.startswith(("rhm_", "phi_", "sub_")))

    components = list(loadings.columns)
    npc = len(components)

    # Wordcloud helper -- mirrors save_wordclouds() but yields an image for inline drawing
    BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
    FONT_PATH = _os.path.join(BASE_DIR, "fonts", f"{font}.ttf")

    def _wc_image(series: pd.Series):
        subdf = series.abs()
        mask = create_dynamic_mask(subdf)

        def _color(word, *args, **kwargs):
            return "#BB0000" if series[word] >= 0 else "#00156A"

        wc = WordCloud(
            font_path=FONT_PATH,
            background_color="white",
            color_func=_color,
            mask=(mask * 255).astype(np.uint8),
            width=mask.shape[1],
            height=mask.shape[0],
            relative_scaling=0.5,
            prefer_horizontal=1000000,
        )
        return wc.generate_from_frequencies(frequencies=subdf.to_dict())

    # Arrange (wordcloud, barplot) pairs in a near-square grid: each pair occupies two
    # grid columns. ncols_pairs is the number of pairs per row.
    ncols_pairs = max(1, int(np.ceil(np.sqrt(npc))))
    nrows_pairs = int(np.ceil(npc / ncols_pairs))
    ncols = ncols_pairs * 2

    pair_w, pair_h = 5.0, 2.8
    fig, axes = plt.subplots(
        nrows_pairs, ncols,
        figsize=(pair_w * ncols_pairs, pair_h * nrows_pairs),
        gridspec_kw={"width_ratios": [1, 1.3] * ncols_pairs},
    )
    axes = np.asarray(axes).reshape(nrows_pairs, ncols)

    for k, pc in enumerate(components):
        pr, pcol = divmod(k, ncols_pairs)
        ax_wc = axes[pr, pcol * 2]
        ax_bar = axes[pr, pcol * 2 + 1]

        ax_wc.imshow(_wc_image(loadings[pc]), interpolation="bilinear")
        ax_wc.set_title(pc, fontsize=12)
        ax_wc.axis("off")

        sub = stats[stats["comp"] == k + 1]
        groups = sub[group].astype(str).tolist()
        vals = sub[col].to_numpy()
        lci = sub[f"{metric}_LCI"].to_numpy()
        uci = sub[f"{metric}_UCI"].to_numpy()
        xerr = np.clip(np.vstack([vals - lci, uci - vals]), 0, None)

        pos = np.arange(len(groups))
        ax_bar.barh(pos, vals, xerr=xerr, capsize=3, color="#4C72B0", ecolor="0.3")
        ax_bar.set_yticks(pos)
        ax_bar.set_yticklabels(groups)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel(labels_for[metric])

        # Bar-centre value labels with leading zero dropped (.83 not 0.83)
        for i, val in zip(pos, vals):
            ax_bar.text(val / 2, i,
                        f"{val:.2f}".replace("-0.", "-.").lstrip("0") or "0",
                        ha="center", va="center", color="white",
                        fontsize=9, fontweight="bold")

    # Hide any leftover cells when npc < ncols_pairs * nrows_pairs (e.g. npc=3 in a 2x2 grid)
    for k in range(npc, nrows_pairs * ncols_pairs):
        pr, pcol = divmod(k, ncols_pairs)
        axes[pr, pcol * 2].axis("off")
        axes[pr, pcol * 2 + 1].axis("off")

    fig.suptitle(title if title is not None else f"By-Component Reproducibility ({labels_for[metric]})",
                 fontsize=13)
    fig.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.6, rect=[0, 0, 1, 0.96])
    return fig
