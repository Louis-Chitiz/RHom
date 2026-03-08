import os
import matplotlib.cm as cm
import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
import numpy as np
from ThoughtSpace.utils import clean_substrings, returnhighest
from wordcloud import WordCloud
import pandas as pd
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFilter


def plot_stats(df, path: str):
    means = df.mean()
    stds = df.std()
    question_names = means.index.tolist()
    max_subst = clean_substrings(question_names)
    if max_subst is not None:
        question_names = [x.replace(max_subst, "") for x in question_names]
        
    plt.bar(question_names,means,yerr=stds)
    plt.title("Average responses")
    plt.xlabel("Item")
    plt.ylabel("Average response")
    plt.xticks(rotation = 45, ha='right')
    plt.tight_layout()
    plt.savefig(path + "_responses.png")
    plt.close()
    print('e')
    pass
    

def plot_scree(pca, path: str):
    """
    Plot the scree plot of the PCA.

    :param pca: The PCA object.
    :param path: The path to save the plot.
    """

    if pca.method == "svd" :
        PC_values = np.arange(pca.fullpca.n_components_) + 1
        expl_var = pca.fullpca.explained_variance_ratio_ * 100
        eigenvals = pca.fullpca.explained_variance_
    
    elif pca.method == "eigen" :
        PC_values = np.arange(len(pca.fullpca)) + 1
        eigenvals =  np.flip(np.sort(pca.fullpca))
        expl_var = (eigenvals / np.sum(pca.eigenvalues)) * 100


    plt.plot(
        PC_values, expl_var, "o-", linewidth=2, color="blue"
        )
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.savefig(path + "_varexp.png")
    plt.close()

    plt.plot(
        PC_values, eigenvals, "o-", linewidth=2, color="blue"
        )
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalues")
    plt.savefig(path + "_eigenvalues.png")
    plt.close()

def create_dynamic_mask(word_freqs, mask_size=600, aspect_ratio = 1.5, blur_radius=15, maskshape='circle', base_intensity=1.5):
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


def save_wordclouds(df: pd.DataFrame, path: str, font: str = "helvetica", n_items_filename: int = 3) -> None:
    """
    This function saves wordclouds to a given path.

    Args:
        df (pd.DataFrame): DataFrame containing the wordclouds.
        path (str): Path to save the wordclouds.
        font (str): Font name for wordclouds.
        n_items_filename (int): Number of items to be included in the filename.

    Returns:
        None
    """
    question_names = df.index.tolist()
    question_names[0] = question_names[0].split("_")[0]
    max_subst = clean_substrings(question_names)
    if max_subst is not None:
        question_names = [x.replace(max_subst, "") for x in question_names]
        df.index = question_names

    for col in df.columns:
        subdf = abs(df[col])

        mask = create_dynamic_mask(subdf)

        def _color(x, *args, **kwargs):
            value = df[col][x]
            if value >= 0:
                # Assign red color for positive values
                return "#BB0000"  # Hex code for red
            else:
                # Assign blue color for negative values
                return "#00156A"  # Hex code for blue

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        FONT_DIR = os.path.join(BASE_DIR, f'fonts\\{font}.ttf')

        wc = WordCloud(
            font_path= FONT_DIR,
            background_color="white",
            color_func=_color,
            mask=(mask * 255).astype(np.uint8),  # Convert mask to correct format
            width=mask.shape[1],
            height=mask.shape[0],
            relative_scaling=0.5,
            prefer_horizontal=1000000,
            # min_font_size=10,
            # max_font_size=200
        )
        df_dict = subdf.to_dict()
        wc = wc.generate_from_frequencies(frequencies=df_dict)
        highest = returnhighest(df[col], n_items_filename)
        plt.figure(figsize = (mask.shape[1]/100, mask.shape[0]/100))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(os.path.join(path, f"{col}_{highest}.png"))
        plt.close()
        # wc.to_file(os.path.join(path, f"{col}_{highest}.png"))

