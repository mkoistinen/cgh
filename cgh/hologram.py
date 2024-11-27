from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from functools import partial
from math import sqrt, cos, sin
from multiprocessing import cpu_count

from numba import cuda
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
    is_cuda_available,
    load_and_transform_mesh,
    process_mesh,
    visualize_wave,
)


DEBUG = False


@cuda.jit
def compute_object_field_cuda(
    X, Y, points, normals, object_field, k, wavelength, object_distance
):
    """
    CUDA kernel to compute the object field contributions.
    Each thread computes the contribution for a single grid point.
    """
    i, j = cuda.grid(2)

    # Ensure thread indices are within bounds
    if i < X.shape[0] and j < X.shape[1]:
        x = X[i, j]
        y = Y[i, j]

        # Initialize complex contribution
        contribution = np.complex64(0.0 + 0.0j)

        for p in range(points.shape[0]):
            # Extract point and normal vectors
            px, py, pz = points[p, 0], points[p, 1], points[p, 2]
            nx, ny, nz = normals[p, 0], normals[p, 1], normals[p, 2]

            # Compute vector from grid point to object point
            dx = np.float32(x - px)
            dy = np.float32(y - py)
            dz = np.float32(pz + object_distance)
            R = sqrt(dx**2 + dy**2 + dz**2)

            if R > 1e-6:  # Avoid division by zero
                # Compute Lambertian scattering
                view_vector = cuda.local.array(3, dtype=np.float32)
                view_vector[0], view_vector[1], view_vector[2] = -dx / R, -dy / R, -dz / R
                normal_vector = cuda.local.array(3, dtype=np.float32)
                normal_vector[0], normal_vector[1], normal_vector[2] = nx, ny, nz

                cos_angle = abs(
                    view_vector[0] * normal_vector[0] +
                    view_vector[1] * normal_vector[1] +
                    view_vector[2] * normal_vector[2]
                )

                if cos_angle > 0:
                    # Accumulate contribution
                    phase = np.float32(k * R)
                    contribution += (cos_angle / R) * (cos(phase) + 1j * sin(phase))

        # Assign contribution to output array
        object_field[i, j] = contribution


def compute_object_field_cuda_driver(points, normals, X, Y, params):
    """
    Driver function to launch the CUDA kernel for object field computation.
    """
    # Parameters
    k = 2 * np.pi / params.wavelength
    object_distance = params.object_distance

    # Transfer data to GPU
    X_device = cuda.to_device(X)
    Y_device = cuda.to_device(Y)
    points_device = cuda.to_device(np.array(points, dtype=np.float32))
    normals_device = cuda.to_device(np.array(normals, dtype=np.float32))
    object_field_device = cuda.device_array(X.shape, dtype=np.complex64)

    # Define CUDA grid and block sizes
    threads_per_block = (16, 16)
    blocks_per_grid_x = (X.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (X.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch CUDA kernel
    compute_object_field_cuda[blocks_per_grid, threads_per_block](
        X_device, Y_device, points_device, normals_device,
        object_field_device, k, params.wavelength, object_distance
    )

    # Transfer result back to CPU
    object_field = object_field_device.copy_to_host()

    return object_field


def compute_object_field_cuda_device(points, normals, params):
    """
    Compute object field using CUDA acceleration.
    """
    # Generate symmetric grid
    X, Y = create_grid(
        params.plate_size,
        params.plate_resolution,
        indexing="xy",
        dtype=np.float32,
    )

    # Call CUDA driver function
    object_field = compute_object_field_cuda_driver(points, normals, X, Y, params)

    return object_field


def compute_chunk_field(
    points_chunk: list[Point3D],
    normals_chunk: list[Point3D],
    X: np.ndarray,
    Y: np.ndarray,
    params: HologramParameters,
    is_chunked: bool = True,
) -> npt.NDArray:
    """
    Compute the wave field contribution for a chunk of points and normals.
    """
    wave_field_chunk = np.zeros_like(X, dtype=np.complex64)
    view_vector = np.array([0, 0, -1], dtype=np.float32)
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
                np.dot(np.array(normal, dtype=np.float32), view_vector),
            )
            if cos_angle <= 0:
                continue

            # Signed differences for proper distance calculation
            dx = X - point.x
            dy = Y - point.y
            dz = point.z + params.object_distance

            # Radial distance
            R = np.sqrt(dx**2 + dy**2 + dz**2)

            # Full complex wave contribution
            wave_contribution = (cos_angle / R) * np.exp(1j * k * R)

            wave_field_chunk += wave_contribution

            if not is_chunked:
                progress.update(task, advance=1)

    return wave_field_chunk


def compute_object_field_cpu(
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


def compute_object_field(
    points: list[Point3D],
    normals: list[Point3D],
    params: HologramParameters,
    num_processes: int = None
):
    """
    Use the best method to compute the object field.
    """
    if is_cuda_available():
        return compute_object_field_cuda_device(points, normals, params)
    else:
        return compute_object_field_cpu(points, normals, params, num_processes)


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
        dtype=np.float32,
    )

    k = 2 * np.pi / params.wavelength

    # Handle planar wave case by setting an effective light source distance
    if params.light_source_distance == 0.0:
        light_source_distance = 1.5e14  # Roughly the distance to the Sun.
    else:
        light_source_distance = params.light_source_distance

    # Compute spherical wave
    R = np.sqrt(X ** 2 + Y ** 2 + light_source_distance ** 2, dtype=np.float32)
    reference_field = np.exp(1j * k * R) / R
    return reference_field


def compute_hologram_old(
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
        dtype=np.float32,
    )

    # Load and process mesh
    mesh_data = load_and_transform_mesh(
        stl_path,
        scale=params.scale_factor,
        rotation=params.rotation_factors,
        translation=params.translation_factors,
        dtype=params.dtype
    )
    points, normals = process_mesh(mesh_data, params.subdivision_factor)

    # Compute wave fields
    if is_cuda_available():
        object_field = compute_object_field_with_cuda(points, normals, params)
    else:
        object_field = compute_object_field_cpu(points, normals, params)
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


def compute_hologram(
    stl_path: Path,
    params: HologramParameters
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Compute a Transmission Hologram using the expanded intensity formula.

    Parameters
    ----------
    stl_path : Path
        Path to STL file.
    params : HologramParameters
        Simulation parameters.

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
        Interference pattern (intensity) and phase information.
    """
    # Generate grid
    X, Y = create_grid(
        params.plate_size,
        params.plate_resolution,
        indexing="xy",
        dtype=params.dtype
    )

    # Load and process mesh
    mesh_data = load_and_transform_mesh(
        stl_path,
        scale=params.scale_factor,
        rotation=params.rotation_factors,
        translation=params.translation_factors,
        dtype=params.dtype
    )
    points, normals = process_mesh(mesh_data, params.subdivision_factor)

    # Compute wave fields
    object_field = compute_object_field(points, normals, params)
    reference_field = compute_reference_field(params)

    # Compute individual intensity terms
    object_intensity = np.abs(object_field) ** 2  # |U_object|^2
    reference_intensity = np.abs(reference_field) ** 2  # |U_reference|^2

    # Compute the interference term
    interference_term = 2 * np.real(object_field * np.conj(reference_field))  # 2 Re(U_object * U_reference*)

    # Combine all terms to form the hologram intensity
    interference_pattern = object_intensity + reference_intensity + interference_term

    # Compute phase for visualization (optional)
    combined_field = object_field + reference_field
    phase = np.angle(combined_field)

    # Debugging and visualization
    if DEBUG:
        print(f"Object Wave Intensity: min={object_intensity.min()}, max={object_intensity.max()}")
        print(f"Reference Wave Intensity: min={reference_intensity.min()}, max={reference_intensity.max()}")
        print(f"Interference Term: min={interference_term.min()}, max={interference_term.max()}")
        print(f"Interference Pattern: min={interference_pattern.min()}, max={interference_pattern.max()}")
        visualize_wave(object_field, params, title="Object Wave")
        visualize_wave(reference_field, params, title="Reference Wave")
        visualize_wave(combined_field, params, title="Combined Wave")

    return interference_pattern, phase
