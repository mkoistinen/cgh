from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from functools import partial
from multiprocessing import cpu_count
import numpy as np
import numpy.typing as npt
from pathlib import Path
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .types import HologramParameters, Point3D
from .utilities import (
    create_grid,
    load_and_scale_mesh,
    process_mesh,
    visualize_wave,
)


DEBUG = False


def compute_reference_field(params: HologramParameters) -> npt.NDArray:
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
    reference_field = np.exp(1j * k * R) / R
    return reference_field


def compute_chunk_field(
    points_chunk: list[Point3D],
    normals_chunk: list[Point3D],
    X: np.ndarray,
    Y: np.ndarray,
    params: HologramParameters,
    is_chunked: True,
) -> npt.NDArray:
    """
    Compute the wave field contribution for a chunk of points and normals.
    """
    wave_field_chunk = np.zeros_like(X, dtype=params.complex_dtype)
    view_vector = np.array([0, 0, -1], dtype=params.dtype)
    k = 2 * np.pi / params.wavelength

    if is_chunked:
        ContextManager = nullcontext()
    else:
        ContextManager = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

    with ContextManager as progress:
        if not is_chunked:
            task = progress.add_task("Computing (at once)", total=len(points_chunk))
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

            if not is_chunked:
                progress.update(task, advance=1)

    return wave_field_chunk


def compute_object_field(
    points: list[Point3D],
    normals: list[Point3D],
    params: HologramParameters,
    num_processes: int = None
) -> npt.NDArray:
    """
    Compute object field using Fresnel approximation with Lambert scattering and occlusion detection.
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
        wave = compute_chunk_field(
            points,
            normals,
            X=X,
            Y=Y,
            params=params,
            is_chunked=False,
        )
        wave_contributions = [wave, ]
    else:
        # Determine chunk size
        num_points = len(points)
        num_chunks = num_processes or cpu_count()
        chunk_size = min(128, (num_points + num_chunks - 1) // num_chunks)

        # Split points and normals into chunks
        chunks = [
            (points[i:i + chunk_size], normals[i:i + chunk_size])
            for i in range(0, num_points, chunk_size)
        ]
        print(f"{len(chunks)=}")

        # Partial function to include fixed arguments
        partial_compute = partial(
            compute_chunk_field,
            X=X, Y=Y, params=params, is_chunked=True
        )

        with (
            ProcessPoolExecutor() as pool,
            Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress,
        ):
            task = progress.add_task("Computing (chunked)", total=len(chunks))
            futures = [
                pool.submit(partial_compute, points_chunk, normals_chunk)
                for points_chunk, normals_chunk in chunks
            ]
            wave_contributions = []
            for future in as_completed(futures):
                wave_contributions.append(future.result())
                progress.update(task, advance=1)

    # Sum contributions from all chunks
    wave_field = np.sum(wave_contributions, axis=0, dtype=params.complex_dtype)

    return wave_field


def compute_hologram(
    stl_path: Path,
    params: HologramParameters
) -> tuple[npt.NDArray, npt.NDArray]:
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
    object_field = compute_object_field(points, normals, params)
    reference_field = compute_reference_field(params)

    scaling_factor = np.abs(object_field).max() / np.abs(reference_field).max() * 0.5
    reference_field *= scaling_factor

    # Compute interference
    combined_field = object_field + reference_field
    phase = np.angle(combined_field)
    interference_pattern = np.abs(combined_field) ** 2

    # Visualize the reference pattern
    if DEBUG:
        for field_type, field in {
            "Object Wave": object_field,
            "Reference Wave": reference_field,
            "Interference Pattern": combined_field
        }.items():
            amplitude = np.abs(field)
            phase = np.angle(field)
            print(f"{field_type} Amplitude: min={amplitude.min()}, max={amplitude.max()}")
            print(f"{field_type} Phase: min={phase.min()}, max={phase.max()}")
            visualize_wave(field, params)

    return interference_pattern, phase
