import numpy as np
import numpy.typing as npt
from pathlib import Path

from .object_field import compute_object_field
from .types import HologramParameters
from .utilities import (
    create_grid,
    load_and_transform_mesh,
    process_mesh,
    Timer,
    visualize_wave,
)


DEBUG = False


def compute_reference_field(params: HologramParameters) -> npt.NDArray:
    """
    Compute a reference wave based on the light source's position defined by `params.reference_field_origin`.

    Parameters
    ----------
    params : HologramParameters
        The hologram parameters, including wavelength, plate size, resolution,
        reference field origin, and other details.

    Returns
    -------
    npt.NDArray
        The computed reference field as a 2D array.
    """
    # Generate the grid
    X, Y = create_grid(
        params.plate_size,
        params.plate_resolution,
        indexing="xy",
        dtype=params.dtype,
    )

    k = 2 * np.pi / params.wavelength
    source_x, source_y, source_z = params.reference_field_origin

    # Compute the distance R from each point on the plate to the reference field origin
    R = np.sqrt((X - source_x)**2 + (Y - source_y)**2 + source_z**2, dtype=params.dtype)

    # Compute the reference field as a spherical wave
    reference_field = np.exp(1j * k * R) / R
    return reference_field


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
    # Load and process mesh
    mesh_data = load_and_transform_mesh(
        stl_path,
        scale=params.scale_factor,
        rotation=params.rotation_factors,
        translation=params.translation_factors,
        dtype=np.float32,
    )
    points, normals = process_mesh(mesh_data, params.subdivision_factor)

    # Compute wave fields
    with Timer("Compute reference field:"):
        reference_field = compute_reference_field(params)
    with Timer("Compute object field:"):
        object_field = compute_object_field(points, normals, params)

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
