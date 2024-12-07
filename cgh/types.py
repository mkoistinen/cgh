from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class Point3D(NamedTuple):
    """3D point representation."""
    x: np.float32
    y: np.float32
    z: np.float32


@dataclass(frozen=True, kw_only=True)
class HologramParameters:
    """
    Parameters for hologram simulation.

    Attributes
    ----------
    wavelength : float
        Wavelength of coherent light in millimeters (e.g., 0.532 for 532nm green laser)
    plate_size : float
        Size of virtual recording plate in millimeters
    plate_resolution : float
        Recording resolution in dots per millimeter
    rotation_factors : tuple[float, float, float], default (0., 0., 0.)
        Rotational transform in degrees about the X, Y, and Z axis.
    translation_factors : tuple[float, float, float], default (0., 0., 0.)
        Translation transform in mm for the X, Y, and Z axis.
    scale_factor : float
        Scale factor for STL object
    subdivision_factor : int
        Number of times to subdivide triangles
    dtype : type
        The float resolution to use.
    complex_dtype : type
        The complex number resolution to use.
    reference_field_origin : tuple[float, float, float]
        The (x, y, z) coordinates of the reference field's origin relative to
        the plate's center.
    illumination_field_origin : tuple[float, float, float]
        The (x, y, z) coordinates of the reference field's origin relative to
        the plate's center.
    """
    complex_dtype: type = np.complex64
    dtype: type = np.float32
    illumination_field_origin: Point3D = 0., 0., -500.0
    plate_resolution: float = 23.622  # dots per mm / 600 dpi
    plate_size: float = 25.4      # 25.4 mm plate
    reference_field_origin: Point3D = 0., 0., 100.0
    rotation_factors: tuple[np.float32, np.float32, np.float32] = (0., 0., 0.)
    scale_factor: float = 12.0
    subdivision_factor: int = 4
    translation_factors: tuple[np.float32, np.float32, np.float32] = (0., 0., 0.)
    use_cuda: bool = False
    wavelength: float = 0.532  # 532 nm (green laser) in mm


class Triangle(NamedTuple):
    """Triangle representation using 3D points."""
    v1: Point3D
    v2: Point3D
    v3: Point3D
