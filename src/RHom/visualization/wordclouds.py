from RHom._deps import pd, np, plt

import os
from typing import Optional
from wordcloud import WordCloud
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw

from RHom.visualization.loadings import returnhighest


def create_dynamic_mask(word_freqs, mask_size=600, aspect_ratio=1.5, blur_radius=15, maskshape='circle', base_intensity=1.5):
    """
    Generates a mask that tightly clusters words while ensuring the largest words are centrally placed.
    """
    num_words = len(word_freqs)

    dominance_ratio = max(word_freqs) / sum(word_freqs)

    # Apply additional scaling based on dominance ratio
    dominance_adjustment = 1.0 - (dominance_ratio * 0.5)  # Reduces size if dominance is high
    mask_size = int(mask_size * dominance_adjustment)

    height = mask_size
    width = int(mask_size * aspect_ratio)

    if maskshape == 'ellipse' or maskshape == 'oval':

        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)

        # Draw a gradient-filled ellipse to prioritize central placement
        for i in range(5, 0, -1):
            alpha = int(255 * (i / 5))  # Gradient levels
            padding = int((1 - (i / 5)) * min(width, height) * 0.5)  # Use min to avoid over-padding

            # Ensure padding does not exceed valid dimensions
            x0, y0 = max(padding, 0), max(padding, 0)
            x1, y1 = min(width - padding, width), min(height - padding, height)

            draw.ellipse((x0, y0, x1, y1), fill=alpha)

        mask = np.array(mask)

    elif maskshape == 'circle':

        dominance_ratio = max(word_freqs) / np.median(word_freqs)

        # Dynamically adjust the mask radius based on the dominance ratio
        base_radius = mask_size // 2.5  # Default central clustering
        adjusted_radius = int(base_radius / np.log1p(dominance_ratio))  # Shrinks as dominance increases

        # Create a dynamically sized circular mask
        mask = np.ones((mask_size, mask_size))
        x, y = np.ogrid[:mask_size, :mask_size]
        center = mask_size // 2
        mask_area = (x - center) ** 2 + (y - center) ** 2 <= adjusted_radius**2
        mask[mask_area] = 0.3  # Words mostly cluster in this area

        # Apply Gaussian blur to create soft edges
        mask = gaussian_filter(mask, sigma=20)

    return mask


def _font_path(font: str) -> str:
    """Resolve a font name to the .ttf path packaged with RHom.visualization."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", f"{font}.ttf")


def make_wordcloud(
    series: pd.Series,
    font: str = "helvetica",
    maskshape: str = "circle",
    pos_color: str = "#BB0000",
    neg_color: str = "#00156A",
    **wc_kwargs,
) -> WordCloud:
    """
    Build a WordCloud from a signed-loading Series.

    Words are sized by ``|loading|`` and coloured by sign: positive loadings use
    ``pos_color``, negative loadings use ``neg_color``. This is the single source of
    truth for the package's wordcloud conventions (mask shape, colour mapping, font
    resolution, default WordCloud kwargs).

    Parameters
    ----------
    series : pd.Series
        Signed loadings indexed by item label. Sign drives colour, ``|value|`` drives size.
    font : str, default "helvetica"
        Font name. Resolves to ``RHom/visualization/fonts/{font}.ttf``.
    maskshape : str, default "circle"
        Forwarded to ``create_dynamic_mask`` -- "circle" or "ellipse"/"oval".
    pos_color, neg_color : str
        Hex colours for positive / negative loadings respectively.
    **wc_kwargs
        Forwarded to ``WordCloud(...)``. Override any default (``background_color``,
        ``relative_scaling``, ``prefer_horizontal``, ``min_font_size``, etc.) or the
        structural arguments (``font_path``, ``mask``, ``width``, ``height``,
        ``color_func``) for advanced use.

    Returns
    -------
    wordcloud.WordCloud
        The generated wordcloud object, ready to pass to ``imshow`` or ``save_wordcloud``.
    """
    abs_series = series.abs()
    mask = create_dynamic_mask(abs_series, maskshape=maskshape)

    def _color(word, *_, **__):
        return pos_color if series[word] >= 0 else neg_color

    wc_args = {
        "font_path": _font_path(font),
        "color_func": _color,
        "mask": (mask * 255).astype(np.uint8),
        "width": mask.shape[1],
        "height": mask.shape[0],
        "background_color": "white",
        "relative_scaling": 0.5,
        "prefer_horizontal": 1000000,
    }
    wc_args.update(wc_kwargs)

    wc = WordCloud(**wc_args)
    return wc.generate_from_frequencies(frequencies=abs_series.to_dict())


def render_wordcloud(
    series: pd.Series,
    ax: Optional[plt.Axes] = None,
    font: str = "helvetica",
    interpolation: str = "bilinear",
    **kwargs,
) -> plt.Axes:
    """
    Build and draw a wordcloud into a matplotlib axes (or a new one).

    Convenience wrapper around ``make_wordcloud`` for plot integration: handles the
    ``imshow`` + ticks-off + spines-off boilerplate that every caller would otherwise
    repeat. Extra kwargs flow through to ``make_wordcloud``.

    Parameters
    ----------
    series : pd.Series
        Signed loadings. See ``make_wordcloud``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. Creates a new figure+axes if None.
    font : str, default "helvetica"
    interpolation : str, default "bilinear"
        Forwarded to ``ax.imshow``.
    **kwargs
        Forwarded to ``make_wordcloud`` (e.g. ``maskshape``, ``pos_color``, ``neg_color``,
        or any ``WordCloud`` kwarg).

    Returns
    -------
    matplotlib.axes.Axes
        The axes the wordcloud was drawn into.
    """
    if ax is None:
        _, ax = plt.subplots()

    wc = make_wordcloud(series, font=font, **kwargs)
    ax.imshow(wc, interpolation=interpolation)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def save_wordcloud(
    series: pd.Series,
    path: str,
    name: Optional[str] = None,
    n_items_filename: int = 3,
    font: str = "helvetica",
    **kwargs,
) -> None:
    """
    Build a single wordcloud and write it to disk as a PNG.

    Filename pattern: ``{name}_{top_n_items}.png`` (or ``{name}.png`` when
    ``n_items_filename`` is 0). ``name`` defaults to ``series.name`` (the column label
    when a DataFrame column is sliced in), so passing a single column straight from a
    loadings frame just works.

    Parameters
    ----------
    series : pd.Series
        Signed loadings to plot.
    path : str
        Directory to write the PNG into. Must exist.
    name : str, optional
        Filename stem. Defaults to ``series.name`` or ``"cloud"``.
    n_items_filename : int, default 3
        Number of top items to encode in the filename via ``returnhighest``. Pass 0
        to skip.
    font : str, default "helvetica"
    **kwargs
        Forwarded to ``make_wordcloud``.
    """
    wc = make_wordcloud(series, font=font, **kwargs)

    stem = name if name is not None else (getattr(series, "name", None) or "cloud")
    if n_items_filename and n_items_filename > 0:
        suffix = returnhighest(series, n_items_filename)
        filename = f"{stem}_{suffix}.png"
    else:
        filename = f"{stem}.png"

    fig = plt.figure(figsize=(wc.width / 100, wc.height / 100))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(os.path.join(path, filename))
    plt.close(fig)


def save_wordclouds(df: pd.DataFrame, path: str, font: str = "helvetica",
                    n_items_filename: int = 3, **kwargs) -> None:
    """
    Save one wordcloud per column of ``df`` to ``path``.

    Backwards-compatible thin wrapper around ``save_wordcloud``. Extra kwargs are
    forwarded to ``make_wordcloud`` (e.g. ``maskshape``, ``pos_color``, ``neg_color``).

    Parameters
    ----------
    df : pd.DataFrame
        Loadings DataFrame: rows are items, columns are components.
    path : str
        Directory to write PNGs into. Must exist.
    font : str, default "helvetica"
    n_items_filename : int, default 3
        Number of top items to encode in each filename.
    """
    for col in df.columns:
        save_wordcloud(df[col], path, name=col,
                       n_items_filename=n_items_filename, font=font, **kwargs)
