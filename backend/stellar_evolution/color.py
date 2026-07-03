"""Blackbody → CIE XYZ → linear sRGB → gamma-encoded sRGB colour pipeline.

This module converts an effective stellar surface temperature (T_eff in K)
into the perceived display colour that a human observer would see when
looking at that star from space through a perfect telescope.

Physical basis
--------------
Stars radiate approximately as blackbodies.  The Planck spectral radiance is:

    B_λ(T) = (2hc²/λ⁵) · 1/(exp(hc/λk_B T) − 1)

Human colour perception is modelled by the CIE 1931 standard observer
colour-matching functions (CMFs) x̄(λ), ȳ(λ), z̄(λ).  Integrating B_λ
against each CMF over the visible spectrum (380–780 nm) gives CIE XYZ
tristimulus values.  These are then converted to display sRGB via the
IEC 61966-2-1 matrix followed by the sRGB gamma encoding.

Pipeline
--------
  T_eff  →  B_λ(T)  →  CIE XYZ  →  linear sRGB  →  gamma sRGB  →  hex / label

References
----------
  - Planck (1901); Wyszecki & Stiles (1982) for CMF tabulation.
  - IEC 61966-2-1 (1999) for the D65 XYZ→sRGB matrix and gamma curve.
  - Massey & Gronwall (1990) for stellar spectral classification vs T_eff.
"""

from __future__ import annotations

import math
from typing import Tuple

from .models import SRGBColor

# ---------------------------------------------------------------------------
# Physical constants (SI units for the Planck function)
# ---------------------------------------------------------------------------

_H = 6.62607015e-34   # Planck constant  [J s]
_C = 2.99792458e8     # Speed of light   [m s⁻¹]
_KB = 1.380649e-23    # Boltzmann const  [J K⁻¹]

# ---------------------------------------------------------------------------
# CIE 1931 standard observer CMFs at 5 nm steps, 380–780 nm (81 points)
#
# Source: Wyszecki & Stiles "Color Science" 2nd ed. (1982), Table 1(3.3.1)
# These are the definitive tabulated values used by colorimetry software
# worldwide.  Each row is (λ nm, x̄, ȳ, z̄).
# ---------------------------------------------------------------------------

_CIE_CMF: Tuple[Tuple[float, float, float, float], ...] = (
    # λ       x̄            ȳ            z̄
    (380, 0.001368, 0.000039, 0.006450),
    (385, 0.002236, 0.000064, 0.010550),
    (390, 0.004243, 0.000120, 0.020050),
    (395, 0.007650, 0.000217, 0.036210),
    (400, 0.014310, 0.000396, 0.067850),
    (405, 0.023190, 0.000640, 0.110200),
    (410, 0.043510, 0.001210, 0.207400),
    (415, 0.077630, 0.002180, 0.371300),
    (420, 0.134380, 0.004000, 0.645600),
    (425, 0.214770, 0.007300, 1.039050),
    (430, 0.283900, 0.011600, 1.385600),
    (435, 0.328500, 0.016840, 1.622960),
    (440, 0.348280, 0.023000, 1.747060),
    (445, 0.348060, 0.029800, 1.782600),
    (450, 0.336200, 0.038000, 1.772110),
    (455, 0.318700, 0.048000, 1.744100),
    (460, 0.290800, 0.060000, 1.669200),
    (465, 0.251100, 0.073900, 1.528100),
    (470, 0.195360, 0.090980, 1.287640),
    (475, 0.142100, 0.112600, 1.041900),
    (480, 0.095640, 0.139020, 0.812950),
    (485, 0.057950, 0.169300, 0.616200),
    (490, 0.032010, 0.208020, 0.465180),
    (495, 0.014700, 0.258600, 0.353300),
    (500, 0.004900, 0.323000, 0.272000),
    (505, 0.002400, 0.407300, 0.212300),
    (510, 0.009300, 0.503000, 0.158200),
    (515, 0.029100, 0.608200, 0.111700),
    (520, 0.063270, 0.710000, 0.078250),
    (525, 0.109600, 0.793200, 0.057250),
    (530, 0.165500, 0.862000, 0.042160),
    (535, 0.225750, 0.914850, 0.029840),
    (540, 0.290400, 0.954000, 0.020300),
    (545, 0.359700, 0.980300, 0.013400),
    (550, 0.433450, 0.994950, 0.008750),
    (555, 0.512050, 1.000000, 0.005750),
    (560, 0.594500, 0.995000, 0.003900),
    (565, 0.678400, 0.978600, 0.002750),
    (570, 0.762100, 0.952000, 0.002100),
    (575, 0.842500, 0.915400, 0.001800),
    (580, 0.916300, 0.870000, 0.001650),
    (585, 0.978600, 0.816300, 0.001400),
    (590, 1.026300, 0.757000, 0.001100),
    (595, 1.056700, 0.694900, 0.001000),
    (600, 1.062200, 0.631000, 0.000800),
    (605, 1.045600, 0.566800, 0.000600),
    (610, 1.002600, 0.503000, 0.000340),
    (615, 0.938400, 0.441200, 0.000240),
    (620, 0.854450, 0.381000, 0.000190),
    (625, 0.751400, 0.321000, 0.000100),
    (630, 0.642400, 0.265000, 0.000050),
    (635, 0.541900, 0.217000, 0.000030),
    (640, 0.447900, 0.175000, 0.000020),
    (645, 0.360800, 0.138200, 0.000010),
    (650, 0.283500, 0.107000, 0.000000),
    (655, 0.218700, 0.081600, 0.000000),
    (660, 0.164900, 0.061000, 0.000000),
    (665, 0.121200, 0.044580, 0.000000),
    (670, 0.087400, 0.032000, 0.000000),
    (675, 0.063600, 0.023200, 0.000000),
    (680, 0.046770, 0.017000, 0.000000),
    (685, 0.032900, 0.011920, 0.000000),
    (690, 0.022700, 0.008210, 0.000000),
    (695, 0.015840, 0.005723, 0.000000),
    (700, 0.011359, 0.004102, 0.000000),
    (705, 0.008111, 0.002929, 0.000000),
    (710, 0.005790, 0.002091, 0.000000),
    (715, 0.004109, 0.001484, 0.000000),
    (720, 0.002899, 0.001047, 0.000000),
    (725, 0.002049, 0.000740, 0.000000),
    (730, 0.001440, 0.000520, 0.000000),
    (735, 0.001000, 0.000361, 0.000000),
    (740, 0.000690, 0.000249, 0.000000),
    (745, 0.000476, 0.000172, 0.000000),
    (750, 0.000332, 0.000120, 0.000000),
    (755, 0.000235, 0.000085, 0.000000),
    (760, 0.000166, 0.000060, 0.000000),
    (765, 0.000117, 0.000042, 0.000000),
    (770, 0.000083, 0.000030, 0.000000),
    (775, 0.000059, 0.000021, 0.000000),
    (780, 0.000042, 0.000015, 0.000000),
)

# Wavelength step (nm → m for Planck integration)
_DELTA_LAMBDA_M: float = 5.0e-9


# ---------------------------------------------------------------------------
# IEC 61966-2-1 D65 XYZ → linear sRGB matrix
# ---------------------------------------------------------------------------

_M_SRGB = (
    ( 3.2406, -1.5372, -0.4986),
    (-0.9689,  1.8758,  0.0415),
    ( 0.0557, -0.2040,  1.0570),
)


# ---------------------------------------------------------------------------
# Planck spectral radiance
# ---------------------------------------------------------------------------


def planck_spectral_radiance(wavelength_m: float, temperature_k: float) -> float:
    """Return the Planck spectral radiance B_λ(T) in W sr⁻¹ m⁻³.

    Parameters
    ----------
    wavelength_m:
        Wavelength in metres.
    temperature_k:
        Blackbody temperature in Kelvin.

    Returns
    -------
    float
        Spectral radiance.  Returns 0 for non-physical inputs.
    """
    if wavelength_m <= 0.0 or temperature_k <= 0.0:
        return 0.0
    exponent = (_H * _C) / (wavelength_m * _KB * temperature_k)
    # Guard against overflow (very short λ or very low T)
    if exponent > 700.0:
        return 0.0
    return (2.0 * _H * _C**2 / wavelength_m**5) / (math.exp(exponent) - 1.0)


# ---------------------------------------------------------------------------
# Blackbody → CIE XYZ
# ---------------------------------------------------------------------------


def blackbody_to_xyz(temperature_k: float) -> Tuple[float, float, float]:
    """Integrate the blackbody spectrum against CIE 1931 CMFs.

    Performs a trapezoidal Riemann sum over the tabulated CIE 1931 standard
    observer CMFs at 5 nm resolution (380–780 nm, 81 points) to produce
    CIE XYZ tristimulus values.  Results are normalised so that Y = 1,
    capturing chromaticity only (luminance is handled separately).

    Parameters
    ----------
    temperature_k:
        Effective surface temperature of the star in Kelvin.

    Returns
    -------
    (X, Y, Z)
        Normalised CIE XYZ tristimulus values with Y = 1.
    """
    temperature_k = max(temperature_k, 100.0)  # floor for safety

    X = Y = Z = 0.0
    for lam_nm, xbar, ybar, zbar in _CIE_CMF:
        lam_m = lam_nm * 1.0e-9
        B = planck_spectral_radiance(lam_m, temperature_k)
        X += B * xbar * _DELTA_LAMBDA_M
        Y += B * ybar * _DELTA_LAMBDA_M
        Z += B * zbar * _DELTA_LAMBDA_M

    # Normalise to Y = 1 to represent chromaticity
    if Y > 0.0:
        X /= Y
        Z /= Y
        Y = 1.0

    return X, Y, Z


# ---------------------------------------------------------------------------
# CIE XYZ → linear sRGB
# ---------------------------------------------------------------------------


def xyz_to_linear_srgb(X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    """Apply the IEC 61966-2-1 D65 matrix to convert XYZ to linear sRGB.

    Parameters
    ----------
    X, Y, Z:
        CIE 1931 XYZ tristimulus values.

    Returns
    -------
    (R_lin, G_lin, B_lin)
        Linear sRGB values (may be outside [0, 1] before clipping).
    """
    m = _M_SRGB
    R_lin = m[0][0] * X + m[0][1] * Y + m[0][2] * Z
    G_lin = m[1][0] * X + m[1][1] * Y + m[1][2] * Z
    B_lin = m[2][0] * X + m[2][1] * Y + m[2][2] * Z
    return R_lin, G_lin, B_lin


# ---------------------------------------------------------------------------
# sRGB gamma encoding  (IEC 61966-2-1)
# ---------------------------------------------------------------------------


def _gamma_encode(linear: float) -> float:
    """Apply the sRGB piecewise gamma transfer function."""
    linear = max(linear, 0.0)
    if linear <= 0.0031308:
        return 12.92 * linear
    return 1.055 * math.pow(linear, 1.0 / 2.4) - 0.055


def apply_srgb_gamma(
    R_lin: float, G_lin: float, B_lin: float
) -> Tuple[float, float, float]:
    """Apply the sRGB gamma transfer function channel-wise.

    Parameters
    ----------
    R_lin, G_lin, B_lin:
        Linear sRGB values.

    Returns
    -------
    (R, G, B)
        Gamma-encoded sRGB values, each clipped to [0.0, 1.0].
    """
    return (
        min(_gamma_encode(R_lin), 1.0),
        min(_gamma_encode(G_lin), 1.0),
        min(_gamma_encode(B_lin), 1.0),
    )


# ---------------------------------------------------------------------------
# Perceptual white-point blend for very hot stars
# ---------------------------------------------------------------------------


def _white_blend_factor(temperature_k: float) -> float:
    """Return how much to blend toward white for very hot stars.

    Above ~10 000 K the blackbody spectrum extends so far into the UV that
    all visible bands are illuminated nearly equally, making the star appear
    brilliant white to human eyes even though the raw chromaticity is blue.
    This blend prevents an unrealistic deep-blue display for hot O/B stars.

    Returns 0.0 at T ≤ 10 000 K (pure blackbody colour) and approaches 0.55
    asymptotically above 40 000 K.
    """
    if temperature_k <= 10000.0:
        return 0.0
    # Smooth sigmoid-like ramp from 10 000 K to 40 000 K
    t = (temperature_k - 10000.0) / 30000.0
    t = min(t, 1.0)
    return 0.55 * (3.0 * t**2 - 2.0 * t**3)  # smoothstep


# ---------------------------------------------------------------------------
# Perceived colour label
# ---------------------------------------------------------------------------


def perceived_label(temperature_k: float) -> str:
    """Return a human-readable colour name for the given effective temperature.

    These ranges are based on the MK spectral classification system and
    standard observational descriptions used in popular astronomy:

    - M class  (< 3900 K)  → deep red / red-orange
    - K class  (3900–5200 K) → orange
    - G class  (5200–6000 K) → yellow-white  (includes the Sun at 5772 K)
    - F class  (6000–7500 K) → white-yellow
    - A class  (7500–10000 K) → white
    - B class  (10000–30000 K) → blue-white
    - O class  (> 30000 K) → deep blue
    """
    t = temperature_k
    if t < 2400:
        return "infrared-red"
    if t < 3200:
        return "deep red"
    if t < 3900:
        return "red-orange"
    if t < 4600:
        return "orange"
    if t < 5200:
        return "orange-yellow"
    if t < 6000:
        return "yellow-white"
    if t < 7500:
        return "white-yellow"
    if t < 10000:
        return "white"
    if t < 20000:
        return "blue-white"
    if t < 35000:
        return "deep blue-white"
    return "deep blue"


# ---------------------------------------------------------------------------
# Full pipeline: blackbody temperature → SRGBColor
# ---------------------------------------------------------------------------


def blackbody_to_srgb(temperature_k: float) -> SRGBColor:
    """Convert an effective stellar temperature to a display-ready sRGB colour.

    This is the single entry-point used by the rest of the backend.
    It runs the full pipeline:

        T_eff → Planck B_λ → CIE XYZ → linear sRGB → white blend
              → sRGB gamma → clip → hex string + label

    Parameters
    ----------
    temperature_k:
        Effective surface temperature in Kelvin.  Values below 100 K are
        clamped to 100 K; values above 100 000 K are accepted but will
        produce near-white results as expected.

    Returns
    -------
    SRGBColor
        All colour representations and the perceived label.

    Examples
    --------
    >>> c = blackbody_to_srgb(5772)   # the Sun
    >>> c.label
    'yellow-white'
    >>> c.srgb[0] > c.srgb[2]         # more red than blue → warm white
    True

    >>> c = blackbody_to_srgb(30000)  # hot B star
    >>> c.label
    'deep blue-white'
    >>> c.srgb[2] >= c.srgb[0]        # more blue than red
    True
    """
    temperature_k = max(float(temperature_k), 100.0)

    # --- Step 1: Planck integration → CIE XYZ (normalised) -----------------
    X, Y, Z = blackbody_to_xyz(temperature_k)

    # Store normalised XYZ for the API response
    xyz_out = [round(X, 6), round(Y, 6), round(Z, 6)]

    # --- Step 2: XYZ → linear sRGB -----------------------------------------
    R_lin, G_lin, B_lin = xyz_to_linear_srgb(X, Y, Z)

    # Clip negative linear values before blending (out-of-gamut blues)
    R_lin = max(R_lin, 0.0)
    G_lin = max(G_lin, 0.0)
    B_lin = max(B_lin, 0.0)

    # Normalise so the brightest channel = 1 (preserve hue at full brightness)
    peak = max(R_lin, G_lin, B_lin, 1.0e-9)
    R_lin /= peak
    G_lin /= peak
    B_lin /= peak

    # Store pre-gamma linear values
    lin_out = [round(R_lin, 6), round(G_lin, 6), round(B_lin, 6)]

    # --- Step 3: Perceptual white blend for very hot stars ------------------
    blend = _white_blend_factor(temperature_k)
    R_lin = R_lin + blend * (1.0 - R_lin)
    G_lin = G_lin + blend * (1.0 - G_lin)
    B_lin = B_lin + blend * (1.0 - B_lin)

    # --- Step 4: sRGB gamma encoding ----------------------------------------
    R, G, B = apply_srgb_gamma(R_lin, G_lin, B_lin)

    # --- Step 5: Round to 3 dp for JSON, build hex --------------------------
    srgb_out = [round(R, 4), round(G, 4), round(B, 4)]

    r8 = int(round(R * 255))
    g8 = int(round(G * 255))
    b8 = int(round(B * 255))
    hex_str = f"#{r8:02x}{g8:02x}{b8:02x}"

    # --- Step 6: Perceived label --------------------------------------------
    label = perceived_label(temperature_k)

    return SRGBColor(
        hex=hex_str,
        srgb=srgb_out,
        linear_srgb=lin_out,
        xyz=xyz_out,
        label=label,
    )
