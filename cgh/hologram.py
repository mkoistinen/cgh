from functools import partial
from multiprocessing import cpu_count, Pool
import numpy as np
import numpy.typing as npt
from pathlib import Path

from . import get_numpy_precision_types
from .types import FloatType, HologramParameters, Point3D
from .utilities import (
    create_grid,
    load_and_scale_mesh,
    process_mesh,
    visualize_wave,
)


FLOAT, COMPLEX = get_numpy_precision_types()

DEBUG = False


def compute_reference_wave_field(params: HologramParameters) -> npt.NDArray:
    """
    Compute a reference wave based on the light source distance.

    If params.light_source_distance == 0.0, treat it as a planar wave, which is
    approximated as coming from an "infinitely" far point light source.
    """
    X, Y = create_grid(
        params.plate_size,
        params.plate_resolution,
        indexing="xy",
        dtype=params.dtype,
    )

    k = 2 * np.pi / params.wavelength

    # Handle planar wave case by setting an effective light source distance
    if params.light_source_distance == 0.0:
        light_source_distance = 1.5e14  # Roughly the distance to the Sun.
    else:
        light_source_distance = params.light_source_distance

    # Compute spherical wave
    R = np.sqrt(X ** 2 + Y ** 2 + light_source_distance ** 2, dtype=params.dtype)
    return np.exp(1j * k * R) / R


def compute_chunk_wave_field(
    points_chunk: list[Point3D],
    normals_chunk: list[Point3D],
    X: np.ndarray,
    Y: np.ndarray,
    params: HologramParameters
) -> npt.NDArray:
    """
    Compute the wave field contribution for a chunk of points and normals.
    """
    wave_field_chunk = np.zeros_like(X, dtype=params.complex_dtype)
    view_vector = np.array([0, 0, -1], dtype=params.dtype)
    k = 2 * np.pi / params.wavelength

    for point, normal in zip(points_chunk, normals_chunk):
        # Lambert scattering
        cos_angle = np.abs(
            np.dot(np.array(normal, dtype=params.dtype), view_vector),
        )
        if cos_angle <= 0:
            continue

        # Signed differences for proper distance calculation
        dx = X - point.x
        dy = Y - point.y
        dz = point.z + params.object_distance

        # Radial distance
        R = np.sqrt(dx**2 + dy**2 + dz**2, dtype=params.dtype)

        # Full complex wave contribution
        wave_contribution = (cos_angle / R) * np.exp(1j * k * R)

        wave_field_chunk += wave_contribution

    return wave_field_chunk


def compute_object_wave_field(
    points: list[Point3D],
    normals: list[Point3D],
    params: HologramParameters,
    num_processes: int = None
) -> npt.NDArray:
    """
    Compute object wave field using Fresnel approximation with Lambert scattering.
    Parallelized implementation using multiprocessing and chunking.
    """
    # Generate symmetric grid
    X, Y = create_grid(
        params.plate_size,
        params.plate_resolution,
        indexing="xy",
        dtype=params.dtype,
    )

    # Use multiprocessing Pool
    if num_processes == 1:
        wave = compute_chunk_wave_field(
            points,
            normals,
            X=X,
            Y=Y,
            params=params,
        )
        wave_contributions = [wave, ]
    else:
        # Determine chunk size
        num_points = len(points)
        num_chunks = num_processes or cpu_count()
        chunk_size = (num_points + num_chunks - 1) // num_chunks

        # Split points and normals into chunks
        chunks = [
            (points[i:i + chunk_size], normals[i:i + chunk_size])
            for i in range(0, num_points, chunk_size)
        ]

        # Partial function to include fixed arguments
        partial_compute = partial(compute_chunk_wave_field, X=X, Y=Y, params=params)

        with Pool(processes=num_processes) as pool:
            wave_contributions = pool.starmap(partial_compute, chunks)

    # Sum contributions from all chunks
    wave_field = np.sum(wave_contributions, axis=0, dtype=params.complex_dtype)

    return wave_field


def compute_hologram(
    stl_path: Path,
    params: HologramParameters
) -> npt.NDArray[FloatType]:
    """
    Compute a Transmission Hologram.

    Parameters
    ----------
    stl_path : Path
        Path to STL file.
    params : HologramParameters
        Simulation parameters.

    Returns
    -------
    npt.NDArray[FloatType]
        Final interference pattern.
    """
    # Generate grid
    X, Y = create_grid(
        params.plate_size,
        params.plate_resolution,
        indexing="xy",
        dtype=params.dtype
    )

    # Load and process mesh
    mesh_data = load_and_scale_mesh(stl_path, params.scale_factor)
    points, normals = process_mesh(mesh_data, params.subdivision_factor)

    # Compute wave fields
    object_wave = compute_object_wave_field(points, normals, params)
    reference_wave = compute_reference_wave_field(params)

    # Visualize the reference pattern
    if DEBUG:
        visualize_wave(reference_wave, params)

    # Compute interference
    combined_wave = object_wave + reference_wave
    interference_pattern = np.abs(combined_wave) ** 2

    print("Object Wave - Max:", np.max(np.abs(object_wave)), "Min:", np.min(np.abs(object_wave)))
    print("Reference Wave - Max:", np.max(np.abs(reference_wave)), "Min:", np.min(np.abs(reference_wave)))
    print("Interference Pattern - Max:", np.max(interference_pattern), "Min:", np.min(interference_pattern))

    return interference_pattern
