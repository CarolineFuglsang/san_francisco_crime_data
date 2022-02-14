import seaborn as sns
import warnings
def custom_palette(n_colors=1, specific_col_idx=None):
    """Returns a custom palette of n_colors (max 11 colors)

        The color palette has been created using `Coolors`_ 
        and added two colors "cinnamon satin (Hex #BD7585)" and "Salmon Pink (Hex: #F193A1)"

        .. _Coolors: https://coolors.co/f94d50-f3722c-f8961e-f9844a-f9c74f-a1c680-3a9278-7ab8b6-577590-206683

        Args:
            n_color: The number of desired colors max 10. Defaults to 1.
            specific_col_idx: list of desired color indexes. Defaults to None.

        Returns: 
            list: List of RGB tuples made as a custom color palette of length `n_colors` or `len(specific_col_idx)` using seaborn.
    """
    colors = ["f94d50", "f3722c", "f8961e", "f9844a", "f9c74f", "a1c680",
              "3a9278", "7ab8b6", "577590", "206683", "F193A1", "B66879"]

    max_colors = len(colors)
    assert n_colors < max_colors, f"n_colors must be less than {max_colors}"

    if specific_col_idx is None:
        if n_colors == 1:
            col_idx = [0]
        if n_colors == 2:
            col_idx = [0, 9]
        if n_colors == 3:
            col_idx = [0, 5, 9]
        if n_colors == 4:
            col_idx = [0, 2, 6, 9]
        if n_colors == 5:
            col_idx = [0, 3, 5, 7, 9]
        if n_colors == 6:
            col_idx = [0, 4, 5, 6, 7, 9]
        if n_colors == 7:
            col_idx = [0, 2, 4, 5, 6, 7, 9]
        if n_colors == 8:
            warnings.warn(
                f"With {n_colors} different colors please be careful that some shades might be close to each other")
            col_idx = [0, 2, 4, 5, 6, 7, 8, 9]
        if n_colors == 9:
            warnings.warn(
                f"With {n_colors} different colors please be careful that some shades might be close to each other")
            col_idx = [0, 2, 3, 4, 5, 6, 7, 8, 9]
        if n_colors == 10:
            warnings.warn(
                f"With {n_colors} different colors please be careful that some shades might be close to each other")
            col_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        col_idx = specific_col_idx
        n_colors = len(col_idx)

    hex_colors = [f'#{colors[col_idx[i]]}' for i in range(n_colors)]

    return sns.color_palette(hex_colors)