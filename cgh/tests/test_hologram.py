from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cgh.hologram import compute_hologram
from cgh.tests.conftest import FLOAT_TYPES
from cgh.types import Point3D, Triangle
from cgh.utilities import subdivide_triangle


DEBUG = False


def test_mesh_processing_accuracy():
    # Create a simple triangle mesh
    triangle = Triangle(
        Point3D(0.0, 0.0, 0.0),
        Point3D(1.0, 0.0, 0.0),
        Point3D(0.0, 1.0, 0.0)
    )
    subdivision_levels = [1, 2, 3, 4]

    for level in subdivision_levels:
        subdivided = subdivide_triangle(triangle, level)
        centroids = [
            Point3D(*(sum(np.float64(v) for v in tri) / 3)) for tri in subdivided
        ]
        print(f"Subdivision Level {level}:")
        for centroid in centroids[:5]:  # Print first 5 centroids
            print(f"Centroid: {centroid}")


@pytest.mark.parametrize("dtype", FLOAT_TYPES)
@pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
@pytest.mark.parametrize("light_source_distance", [0.0, 500.0])
def test_precision_impact(dtype, test_parameters, plate_resolution, light_source_distance):
    """Test the impact of numerical precision on interference pattern symmetry."""
    test_parameters = replace(
        test_parameters,
        plate_resolution=plate_resolution,
        light_source_distance=light_source_distance,
    )
    interference, _phase = compute_hologram(Path("cgh/stls/symmetric_object.stl"), test_parameters)

    # Visualize the interference pattern
    if DEBUG:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(interference, cmap="gray", origin="lower")
        plt.colorbar(label="Intensity")
        plt.title("Interference Pattern")
        plt.xlabel("X (grid units)")
        plt.ylabel("Y (grid units)")
        plt.show()

    # Align shapes for symmetry comparison
    left_half = interference[:, :interference.shape[1] // 2]
    right_half = np.flip(interference[:, (interference.shape[1] + 1) // 2:], axis=1)
    min_columns = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_columns]
    right_half = right_half[:, :min_columns]

    print(f"\n--- Debugging Precision Impact (dtype={dtype}, plate_resolution={plate_resolution}) ---")
    print(f"Interference Pattern Shape: {interference.shape}")
    print(f"Left Half Mean Intensity: {np.mean(left_half):.6f}")
    print(f"Right Half Mean Intensity: {np.mean(right_half):.6f}")
    print(f"Left Half Max Intensity: {np.max(left_half):.6f}")
    print(f"Right Half Max Intensity: {np.max(right_half):.6f}")
    print(f"Max Absolute Difference: {np.max(np.abs(left_half - right_half)):.6f}")
    print(f"Mean Absolute Difference: {np.mean(np.abs(left_half - right_half)):.6f}")

    # Assert symmetry
    np.testing.assert_array_almost_equal(
        left_half, right_half, decimal=6,
        err_msg=f"Interference pattern is not symmetrical at precision {dtype}"
    )


def generate_symmetric_interference(resolution: int, dtype=np.float64) -> np.ndarray:
    """Generate a perfectly symmetric interference pattern for testing."""
    x = np.linspace(-1, 1, resolution, dtype=dtype)
    y = np.linspace(-1, 1, resolution, dtype=dtype)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(2 * np.pi * (X**2 + Y**2))  # Symmetric pattern
    return Z


def generate_symmetric_waves(resolution: int, dtype=np.complex128) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate symmetric object and reference waves.
    """
    x = np.linspace(-1, 1, resolution, dtype=dtype)
    X, Y = np.meshgrid(x, x)

    # Symmetric object wave: centered spherical wave
    object_wave = np.exp(1j * 2 * np.pi * np.sqrt(X**2 + Y**2 + 1)) / (np.sqrt(X**2 + Y**2 + 1))

    # Symmetric reference wave: planar wave propagating in x-direction
    reference_wave = np.exp(1j * 2 * np.pi * X)

    return object_wave, reference_wave


@pytest.mark.parametrize("test_stl_file", ["cgh/stls/symmetric_object.stl"])
def test_full_simulation_basic_properties(test_parameters, test_stl_file):
    """Test basic properties of complete simulation."""
    interference, _phase = compute_hologram(test_stl_file, test_parameters)
    expected_size = int(
        np.round(test_parameters.plate_size * test_parameters.plate_resolution),
    )
    assert interference.shape == (expected_size, expected_size)
    assert np.all(np.isreal(interference))
    assert np.all(interference >= 0)
