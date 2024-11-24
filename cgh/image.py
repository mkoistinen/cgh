from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from PIL import Image


from . import get_numpy_precision_types


FLOAT, COMPLEX = get_numpy_precision_types()


class NormalizationMethod(str, Enum):
    """Enumeration of supported normalization methods."""
    MINMAX = 'minmax'
    LOG = 'log'
    SIGMOID = 'sigmoid'


class OutputType(str, Enum):
    """Enumeration of supported output types."""
    PREVIEW = 'preview'
    HIGH_QUALITY = 'high_quality'


def create_hologram_image(
    interference_pattern: npt.NDArray,
    phase: npt.NDArray,
    output_path: Path,
    output_type: OutputType = OutputType.PREVIEW,
    normalization_method: NormalizationMethod = NormalizationMethod.MINMAX,
    render_type: Literal["phase", "amplitude", "both"] = "phase"
) -> None:
    """
    Convert interference pattern to an image suitable for holographic film.

    Parameters
    ----------
    interference_pattern : npt.NDArray[ComplexType]
        2D array of interference intensities using 64-bit precision
    output_path : Path
        Path to save the output image
    output_type : OutputType, optional
        Type of output image:
        - 'preview': 16-bit PNG
        - 'high_quality': 32-bit float TIFF
    normalization_method : NormalizationMethod, optional
        Method for normalizing intensity values:
        - 'minmax': Linear scaling to full range
        - 'log': Logarithmic scaling
        - 'sigmoid': Sigmoid scaling

    Notes
    -----
    The high_quality output uses TIFF format with floating-point values
    to maintain maximum precision for photographic reproduction.
    """

    phase_normalized = (2 ** 16 - 1) * (phase + np.pi) / (2 * np.pi)  # Map [-pi, pi] to [0, 65535]
    phase_data = phase_normalized.astype(np.uint16)
    phase_img = Image.fromarray(phase_data, mode='I;16')
    phase_img.save("hologram_phase.png", format='PNG')

    # Ensure we're working with 64-bit precision
    pattern = FLOAT(interference_pattern)

    # Normalization
    if normalization_method == NormalizationMethod.MINMAX:
        pattern_min = np.min(pattern)
        pattern_max = np.max(pattern)
        if pattern_max > pattern_min:  # Avoid division by zero
            normalized = (pattern - pattern_min) / (pattern_max - pattern_min)
        else:
            normalized = np.zeros_like(pattern)

    elif normalization_method == NormalizationMethod.LOG:
        # Add small constant to avoid log(0)
        eps = np.finfo(FLOAT).tiny
        normalized = np.log1p(pattern) / np.log1p(np.max(pattern) + eps)

    elif normalization_method == NormalizationMethod.SIGMOID:
        # Center and scale before applying sigmoid
        mean = np.mean(pattern)
        std = np.std(pattern)
        if std > 0:
            normalized = 1 / (1 + np.exp(-(pattern - mean) / std))
        else:
            normalized = np.zeros_like(pattern)
    else:
        raise ValueError(
            f"Unsupported normalization method: {normalization_method}")

    # Output handling
    if output_type == OutputType.PREVIEW:
        # 16-bit PNG
        scaled = (normalized * 65535).astype(np.uint16)
        img = Image.fromarray(scaled, mode='I;16')
        img.save(output_path, format='PNG')

    elif output_type == OutputType.HIGH_QUALITY:
        # 32-bit float TIFF
        # First ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to 32-bit float for TIFF storage
        scaled = normalized.astype(np.float32)

        # Save as TIFF with maximum quality settings
        img = Image.fromarray(scaled, mode='F')
        img.save(
            output_path,
            format='TIFF',
            tiffinfo={
                'compression': 'tiff_deflate',  # Lossless compression
                'resolution_unit': 'RESUNIT_INCH',
                'x_resolution': 11811,  # Example resolution (matches our default)
                'y_resolution': 11811,
                'software': 'Hologram Simulator',
                'description': f'Normalized using {normalization_method.value}'
            }
        )
    else:
        raise ValueError(f"Unsupported output type: {output_type}")
