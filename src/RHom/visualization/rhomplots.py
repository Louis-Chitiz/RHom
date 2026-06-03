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


def plot_dirproj_bypc(results: pd.DataFrame, metric: str = "rhm", title: str = None):
    """
    Per-component grid of pairwise direct-projection heatmaps.

    For each component k the function reconstructs the symmetric G x G pairwise matrix
    from the (referent, comparator, comp=k) rows of `dir_proj_bypc`'s output and plots
    it as a lower-triangle heatmap (diagonal hidden). Panels share a colour scale taken
    from off-diagonal cells across all components, so the magnitude of per-PC pairwise
    similarity is directly comparable across panels.

    Parameters
    ----------
        results: pd.DataFrame
            Output of dir_proj_bypc (one row per (referent, comparator, comp) triple).
        metric: str, default="rhm"
            Which similarity metric to plot: "rhm" (homologue similarity), "phi"
            (factor congruence), or "sub" (subspace similarity).
        title: str, default=None
            Figure-level title. Defaults to the metric's full name.

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
            f"Was dir_proj_bypc run with subspace=True?" if metric == "sub"
            else f"Column '{col}' not found in results."
        )

    components = sorted(results["comp"].unique())
    npc = len(components)
    groups = pd.Index(pd.unique(pd.concat([results["referent"], results["comparator"]])))
    ng = len(groups)

    # Build one symmetric G x G matrix per component
    mats = {}
    for c in components:
        sub = results[results["comp"] == c]
        mtx = pd.DataFrame(1.0, index=groups, columns=groups)
        for _, row in sub.iterrows():
            mtx.loc[row["referent"], row["comparator"]] = row[col]
            mtx.loc[row["comparator"], row["referent"]] = row[col]
        mats[c] = mtx

    # Shared colour scale taken from off-diagonal cells across all panels -- the trivial
    # diagonal (= 1.0) would otherwise squash the dynamic range.
    off_diag = ~np.eye(ng, dtype=bool)
    off_vals = np.concatenate([m.values[off_diag] for m in mats.values()])
    vmin, vmax = float(off_vals.min()), float(off_vals.max())

    ncols = min(npc, 2)
    nrows = int(np.ceil(npc / ncols))
    cell = max(0.6, min(1.0, 10.0 / ng))
    fs = max(7, cell * 13)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * (ng * cell + 2.0) + 1.5, nrows * (ng * cell + 1.5) + 0.6),
        squeeze=False,
    )
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    tri_mask = np.triu(np.ones((ng, ng), dtype=bool))

    for i, c in enumerate(components):
        r, cc = divmod(i, ncols)
        ax = axes[r, cc]
        mtx = mats[c]
        annot = np.vectorize(lambda v: f"{v:.2f}".replace("-0.", "-.").lstrip("0"))(mtx.values)

        sns.heatmap(
            mtx,
            mask=tri_mask,
            vmin=vmin, vmax=vmax,
            annot=annot, fmt="",
            annot_kws={"fontsize": fs},
            cmap="flare",
            square=True,
            linewidths=0.5,
            cbar=(i == 0),
            cbar_ax=(cbar_ax if i == 0 else None),
            cbar_kws={"label": labels_for[metric]} if i == 0 else None,
            ax=ax,
        )
        ax.set_title(f"PC{c}", fontsize=12, pad=6)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fs)

    for i in range(npc, nrows * ncols):
        r, cc = divmod(i, ncols)
        axes[r, cc].set_visible(False)

    fig.suptitle(title if title is not None else f"Per-Component Direct Projection ({labels_for[metric]})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    return fig


def plot_omni_variance(results: pd.DataFrame, group: str = None,
                       n_vars: int = None, title: str = None):
    """
    Stacked bar plot of omnibus variance attribution by group.

    One bar per group; each bar is stacked into npc segments, one per omnibus
    component, with segment heights equal to that component's variance share within
    the group. Stack total = total variance captured by the omnibus solution in
    that group. A dashed horizontal line marks ``npc / n_vars * 100`` -- the share
    a random k-subspace would capture on standardised isotropic data -- to indicate
    when the omnibus components are doing better than chance.

    Parameters
    ----------
        results : pd.DataFrame
            Output of ``omni_variance`` (one row per (group, comp) with ``var_pct``).
        group : str, optional
            Name of the grouping column. Inferred from the layout if not supplied.
        n_vars : int, optional
            Number of decomposition features (p). When provided, the random
            k-subspace chance line is drawn at ``npc / p * 100``%.
        title : str, optional
            Figure-level title.

    Returns
    -------
        matplotlib.figure.Figure
    """
    if group is None:
        reserved = {"n_comp", "comp", "var_pct"}
        group = next(c for c in results.columns if c not in reserved)

    # Pivot to (group × comp) wide form, preserving the input ordering of groups
    group_order = results[group].drop_duplicates().tolist()
    wide = results.pivot(index=group, columns="comp", values="var_pct").reindex(group_order)
    pcs = sorted(wide.columns)
    npc = len(pcs)
    ng = len(group_order)

    fig, ax = plt.subplots(figsize=(max(6.0, ng * 0.9 + 2.0), 5.0))
    colors = sns.color_palette("flare", n_colors=npc)

    bottom = np.zeros(ng)
    for i, pc in enumerate(pcs):
        heights = wide[pc].values
        ax.bar(group_order, heights, bottom=bottom,
               color=colors[i], edgecolor="white", linewidth=0.5,
               label=f"PC{pc}")

        # Per-segment value label, centred inside the segment (skip if too thin)
        for x_i, (h, b) in enumerate(zip(heights, bottom)):
            if h >= 2.0:
                ax.text(x_i, b + h / 2, f"{h:.1f}%",
                        ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        bottom += heights

    # Random k-subspace chance line at npc/p (variance is quadratic in cosine,
    # so the variance baseline is npc/p, not sqrt(npc/p) as for the similarity metrics).
    if n_vars:
        chance = npc / n_vars * 100
        ax.axhline(chance, linestyle="--", color="0.3", linewidth=1.2, zorder=3)
        ax.text(ng - 0.5, chance + 0.6,
                f"random {npc}-subspace ≈ {chance:.1f}%",
                ha="right", va="bottom", color="0.3", fontsize=9)

    # Total-on-top label so the overall reconstructive share is legible
    totals = bottom
    for x_i, total in enumerate(totals):
        ax.text(x_i, total + 0.6, f"{total:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="0.15")

    ax.set_ylabel("% of within-group variance")
    ax.set_ylim(0, max(100.0, float(totals.max()) * 1.1))
    ax.set_xticks(np.arange(ng))
    ax.set_xticklabels(group_order, rotation=30, ha="right")
    ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False)
    ax.set_title(title if title is not None else "Omnibus Variance Attribution by Group",
                 fontsize=13, pad=10)
    fig.tight_layout()
    return fig


def plot_aligned_wordclouds(group_loadings: dict, anchor_loadings=None,
                            font: str = "helvetica", show_var: bool = True,
                            n_features: int = None, title: str = None):
    """
    Grid of wordclouds for per-group component loadings.

    Rows are components (PC1..PC{npc}), columns are groups. When ``anchor_loadings``
    is provided, every group's loadings are Procrustes-rotated to that anchor frame
    before rendering, so column k carries the same homologue identity across columns.

    Parameters
    ----------
        group_loadings : dict[str, pd.DataFrame]
            Mapping ``{group_label: loadings DataFrame (items × PCs)}``. All frames
            must share the same item index and the same number of columns.
        anchor_loadings : pd.DataFrame or array-like, optional
            Reference loadings (items × PCs). When provided, each group's loadings
            are Procrustes-aligned to this frame. When None, loadings are rendered
            as-is.
        font : str, default "helvetica"
            Font name forwarded to ``make_wordcloud``.
        show_var : bool, default True
            If True, label each cell with the variance share captured by that
            component in that group (``sum(L[:, k]**2) / n_features * 100``%).
        n_features : int, optional
            Denominator for variance share. Defaults to the row count of the first
            group's loadings frame.
        title : str, optional
            Figure-level title.

    Returns
    -------
        matplotlib.figure.Figure
    """
    from scipy.linalg import orthogonal_procrustes
    from .wordclouds import make_wordcloud

    groups = list(group_loadings.keys())
    ng = len(groups)
    if ng == 0:
        raise ValueError("group_loadings is empty.")
    first = next(iter(group_loadings.values()))
    items = first.index
    npc = first.shape[1]

    if n_features is None:
        n_features = len(items)

    # Procrustes-align each group's loadings to the shared anchor, if provided
    if anchor_loadings is not None:
        anchor_arr = (anchor_loadings.values if hasattr(anchor_loadings, "values")
                      else np.asarray(anchor_loadings))
        aligned = {}
        for g, L in group_loadings.items():
            L_arr = L.values if hasattr(L, "values") else np.asarray(L)
            R, _ = orthogonal_procrustes(L_arr, anchor_arr)
            aligned[g] = pd.DataFrame(L_arr @ R, index=items,
                                      columns=[f"PC{k+1}" for k in range(npc)])
    else:
        aligned = dict(group_loadings)

    # Two grid rows per PC when variance labels are on: wordcloud + slim label row
    if show_var:
        nrows = 2 * npc
        height_ratios = [4, 1] * npc
    else:
        nrows = npc
        height_ratios = [1] * npc

    fig, axes = plt.subplots(
        nrows, ng,
        figsize=(2.6 * ng, 3.0 * npc),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05},
        squeeze=False,
    )

    for k in range(npc):
        wc_row = (2 * k) if show_var else k
        for c, g in enumerate(groups):
            ax = axes[wc_row, c]
            series = aligned[g].iloc[:, k]
            ax.imshow(make_wordcloud(series, font=font), interpolation="bilinear")
            if wc_row == 0:
                ax.set_title(g, fontsize=12)
            if c == 0:
                ax.set_ylabel(f"PC{k+1}", fontsize=12, rotation=0,
                              labelpad=22, va="center")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if show_var:
                lbl = axes[2 * k + 1, c]
                var_pct = float((series.values ** 2).sum() / n_features * 100)
                lbl.text(0.5, 0.5, f"{var_pct:.1f}% var",
                         transform=lbl.transAxes, ha="center", va="center",
                         fontsize=10, fontweight="bold", color="0.2")
                lbl.axis("off")

    if title:
        fig.suptitle(title, fontsize=13)
        rect = [0, 0, 1, 0.95]
    else:
        rect = [0, 0, 1, 1]
    fig.tight_layout(pad=0.5, rect=rect)
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
    from .wordclouds import make_wordcloud

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

        ax_wc.imshow(make_wordcloud(loadings[pc], font=font), interpolation="bilinear")
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
