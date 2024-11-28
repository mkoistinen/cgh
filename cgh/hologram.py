import numpy as np
import numpy.typing as npt
from pathlib import Path

from .object_field import compute_object_field
from .types import HologramParameters
from .utilities import (
    create_grid,
    load_and_transform_mesh,
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
