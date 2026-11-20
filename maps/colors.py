"""Colors as points: sRGB, Oklab and the hues people sort by.

Oklab is Ottosson's perceptual color space ("A perceptual color space for image
processing", 2020, bottosson.github.io/posts/oklab). Euclidean distance in Oklab is a
reasonable stand-in for how different two colors look, so a color is a point and a
**jump** between two books is the Oklab distance between their colors.
"""

import colorsys
from collections.abc import Sequence

import numpy as np
from matplotlib.colors import CSS4_COLORS

Hex = str  # "#rrggbb"

# Linear sRGB to cone-like responses, and cube-rooted responses to Lab (Ottosson's code).
SRGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
LMS_TO_LAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
# XYZ (D65, white Y = 1) to the same responses, for Ottosson's table of test values.
XYZ_TO_LMS = np.array(
    [
        [0.8189330101, 0.3618667424, -0.1288597137],
        [0.0329845436, 0.9293118715, 0.0361456387],
        [0.0482003018, 0.2643662691, 0.6338517070],
    ]
)


def hex_to_rgb(colors: Sequence[Hex]) -> np.ndarray:
    """sRGB components in [0, 1], one row per "#rrggbb" color."""
    return np.array([[int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)] for c in colors])


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Undo the sRGB transfer curve (IEC 61966-2-1)."""
    rgb = np.asarray(rgb, float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def apply(matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """matrix times each row, written out term by term so that no BLAS kernel (whose
    rounding varies between machines) is involved."""
    rows = np.asarray(rows, float)
    out = np.zeros((len(rows), matrix.shape[0]))
    for j in range(matrix.shape[1]):
        out += rows[:, j, None] * matrix[:, j]
    return out


def lms_to_oklab(lms: np.ndarray) -> np.ndarray:
    """Cube root, then the second matrix."""
    return apply(LMS_TO_LAB, np.cbrt(lms))


def xyz_to_oklab(xyz: np.ndarray) -> np.ndarray:
    """Oklab (L, a, b) of CIE XYZ colors."""
    return lms_to_oklab(apply(XYZ_TO_LMS, xyz))


def oklab(colors: Sequence[Hex]) -> np.ndarray:
    """Oklab (L, a, b) of sRGB colors given as hex strings, one row each."""
    return lms_to_oklab(apply(SRGB_TO_LMS, srgb_to_linear(hex_to_rgb(colors))))


GRAY_CHROMA = 1e-6  # below this chroma a color counts as gray, with hue 0


def oklch(colors: Sequence[Hex]) -> np.ndarray:
    """Lightness, chroma and hue in degrees [0, 360) in Oklab's polar form.

    A gray's a and b are rounding error, so its hue angle would be noise; it gets 0.
    """
    lab = oklab(colors)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
    return np.column_stack([lab[:, 0], chroma, np.where(chroma < GRAY_CHROMA, 0.0, hue)])


def hsv(colors: Sequence[Hex]) -> np.ndarray:
    """Hue in [0, 1), saturation and value, as a color picker shows them."""
    return np.array([colorsys.rgb_to_hsv(*rgb) for rgb in hex_to_rgb(colors)])


def css_colors() -> dict[str, Hex]:
    """The CSS named colors with distinct values, each under its first name alphabetically.

    CSS Color Module Level 4 lists 148 names for 139 colors (gray and grey, aqua and
    cyan, and so on).
    """
    out: dict[str, Hex] = {}
    for name in sorted(CSS4_COLORS):
        value = str(CSS4_COLORS[name]).lower()
        if value not in out.values():
            out[name] = value
    return out


def distances(points: np.ndarray) -> np.ndarray:
    """Matrix of Euclidean distances between rows, without BLAS (bitwise reproducible)."""
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))
