from dataclasses import replace
from functools import lru_cache

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from cgh.hologram import compute_reference_field
from cgh.tests.conftest import FLOAT_TYPES
from cgh.utilities import create_grid


class TestReferenceWave:

    @lru_cache
    def get_reference_wave(self, test_parameters):
        return compute_reference_field(test_parameters)

    @pytest.mark.parametrize("dtype", FLOAT_TYPES)
    @pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
    @pytest.mark.parametrize("light_source_distance", [0.0, 100.0, 500.0])
    def test_reference_wave_symmetry(self, dtype, test_parameters, plate_resolution, light_source_distance):
        """Test the reference wave symmetry both horizontally and vertically."""
        test_parameters = replace(
            test_parameters,
            plate_resolution=plate_resolution,
            wavelength=dtype(test_parameters.wavelength),
            light_source_distance=light_source_distance,
        )
        ref_wave = self.get_reference_wave(test_parameters)

        # Check horizontal symmetry
        left_half = ref_wave[:, :ref_wave.shape[1] // 2]
        right_half = np.flip(ref_wave[:, ref_wave.shape[1] // 2:], axis=1)

        np.testing.assert_array_almost_equal(
            np.abs(left_half), np.abs(right_half), decimal=6,
            err_msg=f"Reference wave is not horizontally symmetrical at precision {dtype}"
        )

        # Check vertical symmetry
        top_half = ref_wave[:ref_wave.shape[0] // 2, :]
        bottom_half = np.flip(ref_wave[ref_wave.shape[0] // 2:, :], axis=0)

        np.testing.assert_array_almost_equal(
            np.abs(top_half), np.abs(bottom_half), decimal=6,
            err_msg=f"Reference wave is not vertically symmetrical at precision {dtype}"
        )

    @pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
    def test_reference_wave_spherical_symmetry(self, plate_resolution, test_parameters):
        """Test reference wave symmetry."""
        test_parameters = replace(
            test_parameters,
            plate_resolution=plate_resolution
        )
        ref_wave = self.get_reference_wave(test_parameters)

        resolution = int(np.round(test_parameters.plate_size * plate_resolution))
        center_x = resolution // 2

        # Split into left and right halves
        left_half = ref_wave[:, :center_x]
        right_half = np.flip(ref_wave[:, center_x + 1:], axis=1)

        # Align shapes for comparison
        min_columns = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_columns]
        right_half = right_half[:, :min_columns]

        # Assert symmetry
        assert_array_almost_equal(
            np.abs(left_half), np.abs(right_half),
            decimal=6, err_msg="Reference wave amplitude is not symmetrical"
        )

    def test_reference_wave_amplitude_falloff(self, test_parameters):
        """Test that reference wave amplitude follows 1/R falloff."""
        ref_wave = self.get_reference_wave(test_parameters)

        X, Y = create_grid(
            test_parameters.plate_size,
            test_parameters.plate_resolution,
            indexing="ij",  # This should be correct here!
        )
        Z_ref = test_parameters.light_source_distance
        R = np.sqrt(X ** 2 + Y ** 2 + Z_ref ** 2)

        # Expected amplitude (1/R)
        expected_amplitude = 1 / R
        actual_amplitude = np.abs(ref_wave)

        # Compare amplitudes
        np.testing.assert_array_almost_equal(
            actual_amplitude, expected_amplitude, decimal=6,
            err_msg="Reference wave amplitude does not follow expected 1/R falloff"
        )
