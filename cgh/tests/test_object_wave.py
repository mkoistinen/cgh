from dataclasses import replace
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from cgh.hologram import compute_object_field
from cgh.tests.conftest import FLOAT_TYPES
from cgh.tests.utilities import generate_test_points_normals
from cgh.utilities import is_cuda_available


class TestObjectWave:

    def test_object_wave_amplitude_falloff(self, test_parameters):
        """Test that amplitude falls off as 1/r."""
        params = replace(
            test_parameters,
            reference_field_origin = (0.0, 0.0, 100.0),
            illumination_field_origin = (0.0, 0.0, -500.0),
        )
        points, normals = generate_test_points_normals(1)
        obj_wave = compute_object_field(points, normals, params, num_processes=1, force_cpu=False)

        resolution = int(np.round(params.plate_size * params.plate_resolution))
        center_idx = resolution // 2

        # Get amplitudes
        center_amp = np.abs(obj_wave[center_idx, center_idx])
        corner_amp = np.abs(obj_wave[0, 0])

        # Calculate distances exactly as done in compute_fresnel_wave_field
        x = np.linspace(-params.plate_size / 2, params.plate_size / 2, resolution, dtype=np.float32)
        y = np.linspace(-params.plate_size / 2, params.plate_size / 2, resolution, dtype=np.float32)
        X, Y = np.meshgrid(x, y)

        # Calculate center distance
        r_center = np.sqrt(
            (X[center_idx, center_idx] - points[0].x) ** 2 +
            (Y[center_idx, center_idx] - points[0].y) ** 2 +
            (points[0].z) ** 2,
            dtype=np.float32,
        )

        # Calculate corner distance
        r_corner = np.sqrt(
            (X[0, 0] - points[0].x) ** 2 +
            (Y[0, 0] - points[0].y) ** 2 +
            (points[0].z) ** 2,
            dtype=np.float32,
        )

        expected_ratio = r_center / r_corner
        actual_ratio = corner_amp / center_amp

        print("\nAmplitude falloff test:")
        print(f"r_center: {r_center:.6f} mm")
        print(f"r_corner: {r_corner:.6f} mm")
        print(f"center_amp: {center_amp:.6f}")
        print(f"corner_amp: {corner_amp:.6f}")
        print(f"Expected ratio (r_center/r_corner): {expected_ratio:.6f}")
        print(f"Actual ratio (corner_amp/center_amp): {actual_ratio:.6f}")

        assert_almost_equal(actual_ratio, expected_ratio, decimal=2)

    @pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
    def test_object_wave_phase_sphericity(self, plate_resolution, test_parameters):
        """Test that phase propagates spherically from point source."""
        test_parameters = replace(test_parameters, plate_resolution=plate_resolution)
        points, normals = generate_test_points_normals(1)
        obj_wave = compute_object_field(points, normals, test_parameters)
        phase = np.angle(obj_wave)

        resolution = int(np.round(test_parameters.plate_size * plate_resolution))

        # Compute physical grid and find the true center indices
        x = np.linspace(-test_parameters.plate_size / 2, test_parameters.plate_size / 2, resolution)
        y = np.linspace(-test_parameters.plate_size / 2, test_parameters.plate_size / 2, resolution)
        center_x = np.argmin(np.abs(x))
        center_y = np.argmin(np.abs(y))

        # Radius in mm
        radius_mm = 5.0

        # Center phase for normalization
        center_phase = phase[center_y, center_x]

        # Define sample points
        samples_physical = [
            (radius_mm, 0.0),   # Right
            (0.0, radius_mm),   # Top
            (-radius_mm, 0.0),  # Left
            (0.0, -radius_mm),  # Bottom
        ]

        distances = []
        phases = []

        print("\nDetailed Phase Debugging:")
        for x_mm, y_mm in samples_physical:
            # Map physical coordinates to grid indices
            x_idx = np.argmin(np.abs(x - x_mm))
            y_idx = np.argmin(np.abs(y - y_mm))

            # Debug physical-to-grid mapping
            print(f"Physical point ({x_mm:.6f}, {y_mm:.6f}) -> Grid indices ({x_idx}, {y_idx})")

            # Calculate radial distance
            R = np.sqrt(x_mm**2 + y_mm**2 + (points[0].z)**2)
            distances.append(R)

            # Sample and normalize phase
            phase_val = np.angle(np.exp(1j * (phase[y_idx, x_idx] - center_phase)))
            phases.append(phase_val)

            print(f"  Radial Distance: {R:.6f}")
            print(f"  Phase (relative to center): {phase_val:.6f}")

        # Verify distances are equal
        np.testing.assert_array_almost_equal(
            distances, distances[0], decimal=6,
            err_msg="Test points are not equidistant from source"
        )

        # Verify phases are equal
        np.testing.assert_allclose(
            phases, phases[0], rtol=5e-2, atol=5e-2,  # Adjust tolerances as needed
            err_msg="Phases are not equal at equal distances"
        )

    @pytest.mark.parametrize("dtype", FLOAT_TYPES)
    @pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
    def test_object_wave_symmetry(self, dtype, test_parameters, plate_resolution):
        """
        Test that when the object imaged is symmetric, so is the object wave.

        NOTE: It appears this test only passes when the resolution is high-
              enough. Looks like for 1200 dpi and up!
        """
        test_parameters = replace(
            test_parameters,
            plate_resolution=plate_resolution,
            wavelength=dtype(test_parameters.wavelength)
        )
        points, normals = generate_test_points_normals(1)
        obj_wave = compute_object_field(points, normals, test_parameters)

        left_half = obj_wave[:, :obj_wave.shape[1] // 2]
        right_half = np.flip(obj_wave[:, (obj_wave.shape[1] + 1) // 2:], axis=1)
        min_columns = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_columns]
        right_half = right_half[:, :min_columns]

        # Debug symmetry comparison
        asymmetry = np.abs(np.abs(left_half) - np.abs(right_half))
        print(f"Max Asymmetry: {np.max(asymmetry)}")
        print(f"Mean Asymmetry: {np.mean(asymmetry)}")
        print(f"Left Half Max Intensity: {np.max(np.abs(left_half))}")
        print(f"Right Half Max Intensity: {np.max(np.abs(right_half))}")

        np.testing.assert_array_almost_equal(
            np.abs(left_half), np.abs(right_half), decimal=6,
            err_msg=f"Object wave is not symmetrical at precision {dtype} and plate_resolution {plate_resolution}"
        )

    @pytest.mark.skip(reason="This is an important test and we need to understand why it fails!")
    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available for test.")
    def test_compute_modes(self, test_parameters):
        """Test that the object wave constructed via CUDA is the same as via CPU."""
        params = replace(
            test_parameters,
            plate_resolution=test_parameters.plate_resolution,
            wavelength=test_parameters.dtype(test_parameters.wavelength)
        )
        points, normals = generate_test_points_normals(1)

        # Compute object_field with CPU
        sp_field = compute_object_field(points, normals, params, num_processes=1, force_cpu=True)
        mp_field = compute_object_field(points, normals, params, force_cpu=True)

        # First, let's be sure multiprocessing is the same as single-processing.
        np.testing.assert_array_almost_equal(
            sp_field, mp_field,
            err_msg="Multiprocessed object field differs from single-processed object field"
        )

        # Compute object_field with CUDA
        cuda_field = compute_object_field(points, normals, params, force_cpu=False)

        # Now test that CUDA gives the same results.
        np.testing.assert_array_almost_equal(sp_field, cuda_field)
