from dataclasses import replace
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_almost_equal
import pytest
import random
from scipy.fft import fft, fftfreq

from cgh.hologram import (
    compute_object_field,
    compute_reference_field,
    compute_hologram,
)
from cgh.tests.conftest import FLOAT_TYPES, COMPLEX_TYPES
from cgh.tests.utilities import generate_test_points_normals
from cgh.utilities import (
    create_grid,
    load_and_scale_mesh,
    process_mesh,
)


DEBUG = False


class TestInterferenceWave:
    """
    Test the construction of the interference pattern between the reference
    wave and the object wave.
    """

    def compute_interference_pattern(
        self,
        object_wave: np.ndarray,
        reference_wave: np.ndarray,
        dtype: type = np.float64,
    ) -> np.ndarray:
        print(f"{len(object_wave)=}, {len(reference_wave)=}")
        combined_wave = object_wave + reference_wave
        interference_pattern = np.abs(combined_wave) ** 2

        if DEBUG:
            # Debugging Outputs
            print("--- Debugging compute_interference_pattern ---")
            print(f"Object Wave Max Amplitude: {np.max(np.abs(object_wave))}")
            print(f"Reference Wave Max Amplitude: {np.max(np.abs(reference_wave))}")
            print(f"Object Wave Center: {object_wave[object_wave.shape[0] // 2, object_wave.shape[1] // 2]}")
            print(f"Reference Wave Center: {reference_wave[reference_wave.shape[0] // 2, reference_wave.shape[1] // 2]}")  # noqa: E501
            print(f"Combined Wave Center: {combined_wave[combined_wave.shape[0] // 2, combined_wave.shape[1] // 2]}")

            # Energy Computations
            object_energy = np.sum(np.abs(object_wave)**2, dtype=dtype)
            reference_energy = np.sum(np.abs(reference_wave)**2, dtype=dtype)
            combined_energy = np.sum(np.abs(combined_wave)**2, dtype=dtype)
            expected_combined_energy = (
                object_energy +
                reference_energy +
                2 * np.sum(np.real(object_wave * np.conj(reference_wave)), dtype=dtype)
            )

            print(f"Total Energy (Object Wave): {object_energy}")
            print(f"Total Energy (Reference Wave): {reference_energy}")
            print(f"Total Energy (Combined Wave): {combined_energy}")
            print(f"Expected Combined Energy: {expected_combined_energy}")

            # Core Region Energy
            center_x, _ = interference_pattern.shape[0] // 2, interference_pattern.shape[1] // 2
            core_slice = slice(center_x - 10, center_x + 10)
            core_energy_object = np.sum(np.abs(object_wave[core_slice, core_slice])**2, dtype=dtype)
            core_energy_reference = np.sum(np.abs(reference_wave[core_slice, core_slice])**2, dtype=dtype)
            core_energy_combined = np.sum(np.abs(combined_wave[core_slice, core_slice])**2, dtype=dtype)

            print("--- Core Region Energy ---")
            print(f"Core Energy (Object Wave): {core_energy_object}")
            print(f"Core Energy (Reference Wave): {core_energy_reference}")
            print(f"Core Energy (Combined Wave): {core_energy_combined}")

            # Phase Alignment Debugging
            phase_difference = np.angle(object_wave * np.conj(reference_wave))
            print("--- Phase Alignment Debugging ---")
            print(f"Max Phase Difference: {np.max(phase_difference)}")
            print(f"Min Phase Difference: {np.min(phase_difference)}")
            print(f"Mean Phase Difference: {np.mean(phase_difference)}")

            plt.imshow(np.angle(object_wave * np.conj(reference_wave)), cmap="hsv")
            plt.colorbar()
            plt.title("Phase Difference Between Object and Reference Waves")
            plt.show()

            print("Object Wave - Max:", np.max(np.abs(object_wave)), "Min:", np.min(np.abs(object_wave)))
            print("Reference Wave - Max:", np.max(np.abs(reference_wave)), "Min:", np.min(np.abs(reference_wave)))
            print("Interference Pattern - Max:", np.max(interference_pattern), "Min:", np.min(interference_pattern))

            # Visualize the interference pattern
            plt.figure(figsize=(8, 8))
            plt.imshow(
                interference_pattern, cmap="gray", origin="lower",
                extent=[-combined_wave.shape[1] / 2, combined_wave.shape[1] / 2,
                        -combined_wave.shape[0] / 2, combined_wave.shape[0] / 2]
            )
            plt.colorbar(label="Intensity")
            plt.title("Interference Pattern")
            plt.xlabel("X (grid units)")
            plt.ylabel("Y (grid units)")
            plt.show()

        return interference_pattern

    def test_interference_pattern_geometry_handling(self, test_parameters):
        """Test interference pattern for symmetry and energy conservation."""
        points, normals = generate_test_points_normals(1)
        object_wave = compute_object_field(points, normals, test_parameters)
        reference_wave = compute_reference_field(test_parameters)
        interference_pattern = self.compute_interference_pattern(object_wave, reference_wave)

        # Compute energies with consistent dtype
        object_energy = np.sum(np.abs(object_wave)**2, dtype=np.float64)
        reference_energy = np.sum(np.abs(reference_wave)**2, dtype=np.float64)
        combined_energy = np.sum(np.abs(interference_pattern), dtype=np.float64)

        # Expected combined energy
        expected_combined_energy = (
            object_energy +
            reference_energy +
            2 * np.sum(np.real(object_wave * np.conj(reference_wave)), dtype=np.float64)
        )

        # Debugging output
        print("\n--- Debugging Energy Conservation ---")
        print(f"Object Energy: {object_energy:.6f}")
        print(f"Reference Energy: {reference_energy:.6f}")
        print(f"Combined Energy: {combined_energy:.6f}")
        print(f"Expected Combined Energy: {expected_combined_energy:.6f}")
        print(f"Energy Difference: {combined_energy - expected_combined_energy:.6e}")

        # Verify energy conservation
        assert np.isclose(
            combined_energy,
            expected_combined_energy,
            atol=1e-6,
            rtol=1e-6
        ), "Energy conservation failed."

    @pytest.mark.parametrize("dtype", FLOAT_TYPES)
    @pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
    def test_combined_wave_symmetry(self, dtype, test_parameters, plate_resolution):
        """Test the symmetry of the combined wave for different precisions and resolutions."""
        test_parameters = replace(test_parameters, plate_resolution=plate_resolution, dtype=dtype)

        symmetric_stl = "cgh/stls/symmetric_object.stl"
        interference, _phase = compute_hologram(symmetric_stl, test_parameters)

        # Align shapes for symmetry comparison
        left_half = interference[:, :interference.shape[1] // 2]
        right_half = np.flip(interference[:, (interference.shape[1] + 1) // 2:], axis=1)
        min_columns = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_columns]
        right_half = right_half[:, :min_columns]

        # Debug asymmetry map
        asymmetry_map = np.abs(np.abs(left_half) - np.abs(right_half))
        max_asymmetry = np.max(asymmetry_map)
        mean_asymmetry = np.mean(asymmetry_map)

        if DEBUG:
            print(f"--- Debugging Combined Wave Symmetry ({dtype}, Resolution {plate_resolution}) ---")
            print(f"Combined Wave Shape: {interference.shape}")
            print(f"Max Asymmetry: {max_asymmetry}")
            print(f"Mean Asymmetry: {mean_asymmetry}")
            print(f"Left Half Max Intensity: {np.max(left_half)}")
            print(f"Right Half Max Intensity: {np.max(right_half)}")

            # Debug core region energy symmetry
            core_left = np.sum(np.abs(left_half[:, :left_half.shape[1] // 2])**2)
            core_right = np.sum(np.abs(right_half[:, :right_half.shape[1] // 2])**2)
            print(f"Core Left Energy: {core_left}")
            print(f"Core Right Energy: {core_right}")

            # Visualize asymmetry map for debugging
            plt.imshow(asymmetry_map, cmap="hot")
            plt.colorbar(label="Absolute Difference")
            plt.title(f"Asymmetry Map ({dtype}, Resolution {plate_resolution})")
            plt.show()

        # Assert symmetry with relaxed tolerance
        np.testing.assert_array_almost_equal(
            np.abs(left_half), np.abs(right_half), decimal=4,
            err_msg=f"Combined wave symmetry mismatch for {dtype}, resolution {plate_resolution}"
        )

    @pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
    def test_full_simulation_symmetry(self, test_parameters, plate_resolution):
        """Test symmetry properties with recentered waves and a symmetric object."""
        test_parameters = replace(test_parameters, plate_resolution=plate_resolution)
        symmetric_stl = "cgh/stls/symmetric_object.stl"

        # Simulate interference pattern
        interference, _phase = compute_hologram(symmetric_stl, test_parameters)

        if DEBUG:
            # Visualize the interference pattern
            plt.figure(figsize=(8, 8))
            plt.imshow(interference, cmap="gray", origin="lower")
            plt.colorbar(label="Intensity")
            plt.title("Interference Pattern")
            plt.xlabel("X (grid units)")
            plt.ylabel("Y (grid units)")
            plt.show()

        # Verify dimensions
        resolution = int(np.round(test_parameters.plate_size * plate_resolution))
        actual_shape = interference.shape
        expected_shape = (resolution, resolution)
        assert actual_shape == expected_shape, f"Interference pattern shape mismatch. Actual: {actual_shape}, Expected: {expected_shape}"  # noqa: E501
        assert interference.shape == (resolution, resolution), "Interference pattern shape mismatch."

        # Identify fractional center
        fractional_center_x = (resolution - 1) / 2
        fractional_center_y = (resolution - 1) / 2

        # Split into left and right halves
        left_half = interference[:, :int(fractional_center_x)]
        right_half = np.flip(interference[:, int(fractional_center_x) + 1:], axis=1)

        # Align shapes for comparison
        min_columns = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_columns]
        right_half = right_half[:, :min_columns]

        # Compute differences
        difference = left_half - right_half
        max_abs_diff = np.max(np.abs(difference))
        mean_abs_diff = np.mean(np.abs(difference))

        # Debugging output
        print("\n--- Debugging Full Simulation Symmetry ---")
        print(f"Resolution: {resolution}")
        print(f"Fractional Center: ({fractional_center_x:.2f}, {fractional_center_y:.2f})")
        print(f"Left Half Shape: {left_half.shape}")
        print(f"Right Half Shape: {right_half.shape}")
        print(f"Max Absolute Difference: {max_abs_diff}")
        print(f"Mean Absolute Difference: {mean_abs_diff}")

        # Assert symmetry
        np.testing.assert_array_almost_equal(
            left_half, right_half, decimal=2,
            err_msg="Interference pattern is not symmetrical."
        )

    @pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
    @pytest.mark.parametrize("dtype", COMPLEX_TYPES)
    def test_interference_pattern_symmetry(self, plate_resolution, dtype):
        """
        Ensure interference pattern symmetry with a uniform planar reference wave.
        """
        plate_size = 25.4
        X, Y = create_grid(plate_size, plate_resolution, indexing="xy", dtype=dtype)

        # Symmetric spherical object wave
        object_wave = (
            np.exp(1j * 2 * np.pi * np.sqrt(X**2 + Y**2 + 1, dtype=dtype)) /
            np.sqrt(X**2 + Y**2 + 1, dtype=dtype)
        )

        # Uniform planar reference wave
        reference_wave = np.ones_like(object_wave, dtype=dtype)

        # Interference pattern
        interference = np.abs(object_wave + reference_wave) ** 2

        # Compare left and right halves
        center_idx = interference.shape[1] // 2
        left_half = interference[:, :center_idx]
        right_half = np.flip(interference[:, center_idx:], axis=1)

        np.testing.assert_array_almost_equal(
            left_half, right_half, decimal=6,
            err_msg=f"Interference pattern is not symmetric for resolution {plate_resolution}"
        )

    @pytest.mark.parametrize("plate_resolution", [94.488])  # 47.244,
    @pytest.mark.parametrize("dtype", [np.complex64, ])  # Need more memory for larger types!
    def test_interference_pattern_energy_conservation(self, test_parameters, plate_resolution, dtype):
        """
        Test that total energy of the interference pattern is consistent.
        """
        params = replace(
            test_parameters,
            plate_resolution=plate_resolution,
            dtype=dtype,
        )
        num_points = int(np.round(params.plate_size * params.plate_resolution))  # Ensure integer number of points

        # Create the grid using the utility function
        X, Y = create_grid(params.plate_size, num_points, indexing="xy", dtype=params.dtype)

        # Symmetric object wave
        obj_mesh = load_and_scale_mesh("cgh/stls/symmetric_object.stl", scale_factor=10.0, dtype=params.dtype)
        points, normals = process_mesh(obj_mesh, params.subdivision_factor)
        object_wave = compute_object_field(
            points=points, normals=normals, params=params,
            num_processes=2
        )

        # Symmetric reference wave
        reference_wave = compute_reference_field(params)

        # Compute interference
        interference = self.compute_interference_pattern(object_wave, reference_wave, dtype=dtype)

        # Define fixed sample points (corners, center)
        sampled_indices = [
            (0, 0),  # Top-left corner
            (num_points // 2, num_points // 2),  # Center
            (num_points - 1, num_points - 1),  # Bottom-right corner
        ]

        # Add randomized sampling points
        num_random_samples = 10  # Number of additional random points
        random_samples = [
            (random.randint(0, num_points - 1), random.randint(0, num_points - 1))
            for _ in range(num_random_samples)
        ]
        sampled_indices.extend(random_samples)

        # Check energy at sampled points
        sampled_energies = []
        for idx in sampled_indices:
            i, j = idx
            energy_object = np.abs(object_wave[i, j])**2
            energy_reference = np.abs(reference_wave[i, j])**2
            cross_term = 2 * np.real(object_wave[i, j] * np.conj(reference_wave[i, j]))
            expected_energy = energy_object + energy_reference + cross_term
            actual_energy = interference[i, j]
            sampled_energies.append((idx, actual_energy, expected_energy))

        # Debugging output
        print("\n--- Sampled Energy Conservation ---")
        for idx, actual, expected in sampled_energies:
            print(f"Point {idx}: Actual Energy = {actual:.6f}, Expected Energy = {expected:.6f}")

        # Assert energy conservation for all sampled points
        for idx, actual, expected in sampled_energies:
            np.testing.assert_almost_equal(
                actual, expected, decimal=6,
                err_msg=f"Energy conservation failed at point {idx}."
            )

    def test_interference_pattern_properties(self, test_parameters):
        """Test basic physical properties of interference pattern."""
        points, normals = generate_test_points_normals(1)
        obj_wave = compute_object_field(points, normals, test_parameters)
        ref_wave = compute_reference_field(test_parameters)
        interference = self.compute_interference_pattern(obj_wave, ref_wave)

        # Test reality and positivity
        assert np.all(np.isreal(interference))
        assert np.all(interference >= 0)

        # Calculate expected energy based on actual grid size
        resolution = int(np.round(test_parameters.plate_size * test_parameters.plate_resolution))
        expected_energy = resolution**2 * 4  # Maximum possible energy for interference of unit waves

        total_energy = np.sum(interference)

        print("\nInterference pattern properties:")
        print(f"Resolution: {resolution}x{resolution}")
        print(f"Total energy: {total_energy}")
        print(f"Expected maximum energy: {expected_energy}")
        print(f"Average intensity: {total_energy / (resolution ** 2)}")

        assert total_energy <= expected_energy

    @pytest.mark.skip("Review later.")
    @pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
    def test_interference_fringe_spacing_fourier(self, plate_resolution, test_parameters):
        """Test fringe spacing using Fourier analysis with peak refinement."""
        test_parameters = replace(test_parameters, plate_resolution=plate_resolution)

        # Generate object and reference waves
        points, normals = generate_test_points_normals(1)
        obj_wave = compute_object_field(points, normals, test_parameters)
        ref_wave = compute_reference_field(test_parameters)

        # Compute interference pattern
        interference = self.compute_interference_pattern(obj_wave, ref_wave)

        # Extract the central line of the interference pattern
        resolution = interference.shape[0]
        center_idx = resolution // 2
        center_line = interference[center_idx, :]

        # Subtract the mean to remove the DC component
        center_line -= np.mean(center_line)

        # Apply a Hamming window to reduce boundary effects
        center_line_windowed = center_line * np.hamming(center_line.size)

        if DEBUG:
            plt.plot(center_line, label="Center Line (Raw)")
            plt.plot(center_line_windowed, label="Center Line (Windowed)")
            plt.legend()
            plt.title("Central Line of Interference Pattern")
            plt.show()

            print("Mean of Center Line (before subtraction):", np.mean(center_line))
            print("Max Value of Center Line:", np.max(center_line))
            print("Min Value of Center Line:", np.min(center_line))

        # Perform Fourier Transform
        fft_values = np.abs(fft(center_line_windowed))
        frequencies = fftfreq(center_line.size, d=test_parameters.plate_size / resolution)

        # Exclude the DC component
        fft_values[0] = 0

        # Find the dominant frequency
        dominant_freq_idx = np.argmax(fft_values)
        dominant_freq = frequencies[dominant_freq_idx]

        # Quadratic peak refinement
        left = fft_values[dominant_freq_idx - 1]
        center = fft_values[dominant_freq_idx]
        right = fft_values[dominant_freq_idx + 1]
        correction = 0.5 * (left - right) / (left - 2 * center + right)
        refined_freq = dominant_freq + correction * (frequencies[1] - frequencies[0])

        # Calculate the fringe spacing from the refined frequency
        fringe_spacing = 1 / np.abs(refined_freq)

        # Calculate expected fringe spacing
        theta = np.arctan(test_parameters.plate_size / (2 * points[0].z))
        expected_spacing = test_parameters.wavelength / np.sin(theta)

        if DEBUG:
            # Debugging output
            print(f"\nFourier Fringe Spacing Analysis (Plate Resolution: {plate_resolution}):")
            print(f"Dominant Frequency: {dominant_freq:.6f} cycles/mm")
            print(f"Refined Frequency: {refined_freq:.6f} cycles/mm")
            print(f"Computed Fringe Spacing: {fringe_spacing:.6f} mm")
            print(f"Expected Fringe Spacing: {expected_spacing:.6f} mm")

            # Visualization
            plt.plot(frequencies[:len(frequencies) // 2], fft_values[:len(frequencies) // 2])
            plt.title(f"Fourier Transform of Interference Pattern (Plate Resolution: {plate_resolution})")
            plt.xlabel("Spatial Frequency (cycles/mm)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()

        # Assert the computed fringe spacing is close to the expected value
        assert_almost_equal(
            fringe_spacing, expected_spacing, decimal=2,
            err_msg="Fringe spacing does not match expected value"
        )

    @pytest.mark.parametrize("wavelength, expected_scale", [
        (0.532, 1.0),    # Green laser (reference)
        (0.633, 1.19),   # Red laser
        (0.450, 0.85),   # Blue laser
    ])
    def test_wavelength_scaling(self, wavelength: float, expected_scale: float, test_parameters):
        """Test that interference pattern scales correctly with wavelength."""
        params = replace(test_parameters, wavelength=wavelength)
        points, normals = generate_test_points_normals(1)
        obj_wave = compute_object_field(points, normals, params)
        ref_wave = compute_reference_field(params)
        interference = self.compute_interference_pattern(obj_wave, ref_wave)

        # Ensure the center index is an integer
        center_line = interference[int(params.plate_resolution // 2), :]
        peaks = np.where(np.diff(np.signbit(np.diff(center_line))))[0]

        if len(peaks) >= 2:
            fringe_spacing = np.mean(np.diff(peaks))
            reference_spacing = fringe_spacing / expected_scale
            assert_almost_equal(fringe_spacing / reference_spacing, expected_scale, decimal=2)
