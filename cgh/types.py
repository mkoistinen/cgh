from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
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
    light_source_distance : float
        Distance of coherent source in millimeters, use 0.0 for planar wave.
    object_distance : float
        Distance of object behind plate in millimeters
    scale_factor : float
        Scale factor for STL object
    subdivision_factor : int
        Number of times to subdivide triangles
    dtype : type
        The float resolution to use.
    complex_dtype : type
        The complex number resolution to use.
    """
    wavelength: float = 0.532  # 532 nm (green laser) in mm
    plate_size: float = 25.4      # 25.4 mm plate
    plate_resolution: float = 23.622  # dots per mm / 600 dpi
    light_source_distance: float = 500.0  # mm
    object_distance: float = 50.0  # mm behind plate
    scale_factor: float = 12.0
    subdivision_factor: int = 4
    # TODO: Rename this parameter to something more useful and make sure it is used everywhere.
    dtype: type = np.float64
    complex_dtype: type = np.complex128


class Point3D(NamedTuple):
    """3D point representation."""
    x: np.float64
    y: np.float64
    z: np.float64


class Triangle(NamedTuple):
    """Triangle representation using 3D points."""
    v1: Point3D
    v2: Point3D
    v3: Point3D
