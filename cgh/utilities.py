from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import typing as t
import warnings

import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
import numpy.typing as npt
from stl import mesh

from .types import HologramParameters, Point3D, Triangle


class Timer:
    """
    A context manager that records the duration of the execution of the body.

    Parameters
    ----------
    msg : str, default ""
        If non-empty, prints the string on exit followed by the duration.
    """

    def __init__(self, msg=""):  # type: ignore reportMissingSuperCall
        self._msg = msg
        self._start = None
        self._end = None
        self.duration = timedelta(0)

    def __enter__(self) -> "Timer":
        self._start = datetime.now()
        return self

    def __exit__(self, *args):
        self.duration: timedelta = datetime.now() - self._start
        if self._msg:
            print(self._msg, self.duration)


def load_and_transform_mesh(
    stl_path: Path,
    *,
    rotation: t.Optional[tuple[np.float32, np.float32, np.float32]] = None,
    translation: tuple[np.float32, np.float32, np.float32] = None,
    scale: float,
    dtype: float = np.float32,
) -> npt.NDArray[np.float32]:
    """
    Load, scale, rotate (in degrees), and translate an STL mesh.

    Parameters
    ----------
    stl_path : Path
        Path to STL file.
    rotation: tuple[np.float32, np.float32, np.float32]
        Rotation angles (in degrees) about the X, Y, and Z axes, respectively.
    translation: tuple[np.float32, np.float32, np.float32]
        Translation offsets along the X, Y, and Z axes, respectively.
    scale : float
        Scale factor to apply to mesh about the object's local origin.
    dtype : type, default np.float32
        The float precision to use.

    Returns
    -------
    npt.NDArray[np.float32]
        Transformed mesh vertices.
    """
    if rotation is None:
        rotation = 0., 0., 0.

    if translation is None:
        translation = 0., 0., 0.

    # Load the mesh
    stl_mesh = mesh.Mesh.from_file(str(stl_path))
    vertices = dtype(stl_mesh.vectors.reshape(-1, 3))  # Flatten triangles into vertices

    # Scale the vertices
    vertices *= scale

    # Rotation matrix
    def rotation_matrix(rx_deg, ry_deg, rz_deg):
        # Convert degrees to radians
        rx = np.radians(rx_deg)
        ry = np.radians(ry_deg)
        rz = np.radians(rz_deg)

        # Rotation about X-axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)],
        ], dtype=dtype)

        # Rotation about Y-axis
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ], dtype=dtype)

        # Rotation about Z-axis
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1],
        ], dtype=dtype)

        return Rz @ Ry @ Rx

    # Apply rotation
    R = rotation_matrix(*rotation)
    vertices = vertices @ R.T

    # Apply translation
    translation_vector = np.array(translation, dtype=dtype)
    vertices += translation_vector

    # Reshape back into triangular form
    transformed_vectors = vertices.reshape(-1, 3, 3)

    return transformed_vectors


def compute_triangle_normal(
    triangle: Triangle,
    dtype: float = np.float32
) -> Point3D:
    """
    Compute normal vector for a triangle.

    Parameters
    ----------
    triangle : Triangle
        Input triangle
    dtype : type, default np.float32
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
    mesh_data: npt.NDArray[np.float32],
    subdivision_factor: int,
    dtype: type = np.float32,
) -> tuple[list[Point3D], list[Point3D]]:
    """
    Process mesh data into centroids (points) and normals.

    Parameters
    ----------
    mesh_data : npt.NDArray[FloatType]
        Raw mesh data
    subdivision_factor : int
        Subdivision factor
    dtype : type, default np.float32
        The float precision to use.

    Returns
    -------
    tuple[list[Point3D], list[Point3D]]
        Triangle centroids (points) and their corresponding normals
    """
    with Timer() as timer:
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

            subdivided = subdivide_triangle(
                triangle,
                subdivision_factor,
                dtype=dtype,
            )
            for sub_triangle in subdivided:
                # Compute centroid
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
                    centroid = Point3D(*(sum(np.float32(p) for p in sub_triangle) / 3))
                points.append(centroid)
                normals.append(normal)

    print(f"Subdividing mesh into {len(points):,d} points required: {timer.duration}")
    return points, normals


@lru_cache
def create_grid(
    size: float,
    resolution: float,
    indexing: str = 'ij',
    dtype: type = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
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
    dtype : np.dtype, default np.float32
        The precision float type for the grid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
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
    levels: int,
    dtype: float = np.float32,
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
        mid1 = Point3D(*(dtype(triangle.v1) + dtype(triangle.v2)) / 2)
        mid2 = Point3D(*(dtype(triangle.v2) + dtype(triangle.v3)) / 2)
        mid3 = Point3D(*(dtype(triangle.v3) + dtype(triangle.v1)) / 2)

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


def show_grid_memory_requirements(params: HologramParameters):
    X, Y = create_grid(
        params.plate_size,
        params.plate_resolution,
        indexing="xy",
        dtype=params.dtype,
    )
    bytes = X.nbytes + Y.nbytes
    print(f"Grid requires {bytes:,d} bytes ({bytes / 1024 ** 2:,.3f} MB)")


@lru_cache
def is_cuda_available() -> bool:
    try:
        # Check if CUDA is available
        cuda.get_current_device()
        return True
    except cuda.CudaSupportError:
        return False
