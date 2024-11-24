from pathlib import Path
from typing import Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from stl import mesh

from . import get_numpy_precision_types
from .types import FloatType, HologramParameters, Point3D, Triangle


FLOAT, COMPLEX = get_numpy_precision_types()


def load_and_scale_mesh(
    stl_path: Path,
    scale_factor: float,
    dtype: type = np.float64,
) -> npt.NDArray[FloatType]:
    """
    Load and scale an STL mesh.

    Parameters
    ----------
    stl_path : Path
        Path to STL file
    scale_factor : float
        Scale factor to apply to mesh
    dtype : type, default np.float64
        The float precision to use.

    Returns
    -------
    npt.NDArray[np.FloatType]
        Scaled mesh vertices
    """
    stl_mesh = mesh.Mesh.from_file(str(stl_path))
    return dtype(stl_mesh.vectors) * scale_factor


def compute_triangle_normal(triangle: Triangle, dtype: type = np.float64) -> Point3D:
    """
    Compute normal vector for a triangle.

    Parameters
    ----------
    triangle : Triangle
        Input triangle
    dtype : type, default np.float64
        The float precision to use.

    Returns
    -------
    Point3D
        Normalized normal vector
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        v1 = np.array([triangle.v2.x - triangle.v1.x,
                       triangle.v2.y - triangle.v1.y,
                       triangle.v2.z - triangle.v1.z], dtype=dtype)
        v2 = np.array([triangle.v3.x - triangle.v1.x,
                       triangle.v3.y - triangle.v1.y,
                       triangle.v3.z - triangle.v1.z], dtype=dtype)
    normal = np.cross(v1, v2)
    length = np.sqrt(np.sum(normal**2, dtype=dtype), dtype=dtype)
    if length > 0:
        normal = normal / length
    return Point3D(*normal)


def process_mesh(
    mesh_data: npt.NDArray[FloatType],
    subdivision_factor: int,
    dtype: type = np.float64,
) -> Tuple[list[Point3D], list[Point3D]]:
    """
    Process mesh data into points and normals.

    Parameters
    ----------
    mesh_data : npt.NDArray[FloatType]
        Raw mesh data
    subdivision_factor : int
        Subdivision factor
    dtype : type, default np.float64
        The float precision to use.

    Returns
    -------
    Tuple[List[Point3D], List[Point3D]]
        Points and their corresponding normals
    """
    points = []
    normals = []

    for triangle_vertices in mesh_data:
        triangle = Triangle(
            Point3D(*triangle_vertices[0]),
            Point3D(*triangle_vertices[1]),
            Point3D(*triangle_vertices[2]),
        )
        normal = compute_triangle_normal(triangle, dtype=dtype)

        # Skip triangles with invalid normals
        if np.all(np.array(normal, dtype=dtype) == 0):
            continue

        subdivided = subdivide_triangle(triangle, subdivision_factor)
        for sub_triangle in subdivided:
            # Compute centroid
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
                centroid = Point3D(*(sum(FLOAT(p) for p in sub_triangle) / 3))
            points.append(centroid)
            normals.append(normal)

    return points, normals


def create_grid(
    size: float,
    resolution: float,
    indexing: str = 'ij',
    dtype: type = np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D grid of coordinates centered at (0, 0).

    Parameters
    ----------
    size : float
        Physical size of the grid (in mm).
    resolution : float
        Number of dots per mm (floating-point).
    indexing : str
        Meshgrid indexing ('ij' for row-major or 'xy' for Cartesian).
    dtype : np.dtype, default np.float64
        The precision float type for the grid.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Meshgrid for X and Y coordinates.
    """
    # Calculate grid points
    grid_points = int(np.round(size * resolution))  # Ensure integer grid points
    x = np.linspace(-size / 2, size / 2, grid_points, dtype=dtype)
    y = np.linspace(-size / 2, size / 2, grid_points, dtype=dtype)

    # Create meshgrid
    X, Y = np.meshgrid(x, y, indexing=indexing)

    return X, Y


def subdivide_triangle(
    triangle: Triangle,
    levels: int
) -> list[Triangle]:
    """
    Recursively subdivide a triangle.

    Parameters
    ----------
    triangle : Triangle
        Triangle to subdivide
    levels : int
        Number of subdivision levels

    Returns
    -------
    List[Triangle]
        List of subdivided triangles
    """
    if levels == 0:
        return [triangle]

    # Compute midpoints with 128-bit precision
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        mid1 = Point3D(*(FLOAT(triangle.v1) + FLOAT(triangle.v2)) / 2)
        mid2 = Point3D(*(FLOAT(triangle.v2) + FLOAT(triangle.v3)) / 2)
        mid3 = Point3D(*(FLOAT(triangle.v3) + FLOAT(triangle.v1)) / 2)

    new_triangles = [
        Triangle(triangle.v1, mid1, mid3),
        Triangle(mid1, triangle.v2, mid2),
        Triangle(mid3, mid2, triangle.v3),
        Triangle(mid1, mid2, mid3)
    ]

    result = []
    for new_triangle in new_triangles:
        result.extend(subdivide_triangle(new_triangle, levels - 1))
    return result


def visualize_wave(reference_wave: np.ndarray, params: HologramParameters) -> None:
    """
    Visualize components of a complex wave.

    Parameters
    ----------
    reference_wave : np.ndarray
        Complex-valued reference wave.
    params : HologramParameters
        Simulation parameters for visualization metadata.
    """
    amplitude = np.abs(reference_wave)
    phase = np.angle(reference_wave)
    real_part = np.real(reference_wave)
    imag_part = np.imag(reference_wave)

    plt.figure(figsize=(10, 8))

    # Amplitude
    plt.subplot(2, 2, 1)
    plt.title("Amplitude")
    plt.imshow(amplitude, cmap="viridis", extent=(-params.plate_size / 2, params.plate_size / 2,
                                                  -params.plate_size / 2, params.plate_size / 2))
    plt.colorbar(label="Amplitude")

    # Phase
    plt.subplot(2, 2, 2)
    plt.title("Phase")
    plt.imshow(phase, cmap="twilight", extent=(-params.plate_size / 2, params.plate_size / 2,
                                               -params.plate_size / 2, params.plate_size / 2))
    plt.colorbar(label="Phase (radians)")

    # Real Part
    plt.subplot(2, 2, 3)
    plt.title("Real Part")
    plt.imshow(real_part, cmap="coolwarm", extent=(-params.plate_size / 2, params.plate_size / 2,
                                                   -params.plate_size / 2, params.plate_size / 2))
    plt.colorbar(label="Real Part")

    # Imaginary Part
    plt.subplot(2, 2, 4)
    plt.title("Imaginary Part")
    plt.imshow(imag_part, cmap="coolwarm", extent=(-params.plate_size / 2, params.plate_size / 2,
                                                   -params.plate_size / 2, params.plate_size / 2))
    plt.colorbar(label="Imaginary Part")

    plt.tight_layout()
    plt.show()
