from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from functools import partial
import math
from multiprocessing import cpu_count

from numba import cuda
import numpy as np
import numpy as np
import numpy.typing as npt
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
)


def compute_object_field(
    points: list[Point3D],
    normals: list[Point3D],
    params: HologramParameters,
    num_processes: int = None,
    force_cpu: bool = False
):
    """
    Use the best method to compute the object field.
    """
    if not force_cpu and is_cuda_available():
        return compute_object_field_cuda_device(points, normals, params)
    else:
        return compute_object_field_cpu(points, normals, params, num_processes)


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
    # object_field_device = cuda.device_array_like(X)
    object_field_device = cuda.device_array(X.shape, dtype=np.complex64)  # Corrected


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

            # Radial distance
            R = math.sqrt(dx**2 + dy**2 + dz**2)

            if R > 1e-9:  # Avoid division by zero
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
                    contribution += (cos_angle / R) * (math.cos(phase) + 1j * math.sin(phase))

        # Assign contribution to output array
        object_field[i, j] = contribution


def compute_object_field_cpu(
    points: list[Point3D],
    normals: list[Point3D],
    params: HologramParameters,
    num_processes: int = None,
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
        dtype=np.float32,
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
    wave_field = np.sum(wave_contributions, axis=0, dtype=np.complex64)

    return wave_field


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

    This is done on CPU and uses multiprocessing. This is the worker function.
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

            if cos_angle <= 0.0:
                continue

            # Signed differences for proper distance calculation
            dx = X - point.x
            dy = Y - point.y
            dz = point.z + params.object_distance

            # Radial distance
            R = np.sqrt(dx**2 + dy**2 + dz**2)

            # Full complex wave contribution
            wave_contribution = (cos_angle / R) * np.exp(1j * np.float32(k * R))
            wave_field_chunk += wave_contribution

            if not is_chunked:
                progress.update(task, advance=1)

    return wave_field_chunk
