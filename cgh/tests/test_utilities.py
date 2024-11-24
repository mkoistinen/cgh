import numpy as np
import pytest

from cgh.utilities import create_grid


@pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
def test_grid_symmetry(plate_resolution):
    # TODO: This should handle non-square plate_sizes
    size = 25.4
    X, Y = create_grid(size, plate_resolution, indexing="ij")

    # Assert grid shape symmetry
    assert X.shape == Y.shape, "Grid shapes do not match."

    if len(X) % 2 == 0 and len(Y) % 2 == 0:
        c_x = X.shape[0] // 2 - 1
        c_y = Y.shape[0] // 2 - 1
        assert np.isclose(X[c_x, c_y], -X[c_x + 1, c_y], atol=1e-6), "Center values are not symmetrical."
    elif len(X) % 2 and len(Y) % 2:
        c_x, c_y = X.shape[0] // 2, Y.shape[1] // 2
        # Assert central symmetry
        assert np.isclose(X[c_x, c_y], 0, atol=1e-6), "Center value of X grid is not zero."
        assert np.isclose(Y[c_x, c_y], 0, atol=1e-6), "Center value of Y grid is not zero."

    # Assert left-right and top-bottom symmetry using absolute values
    assert np.allclose(np.abs(X[:, 0]), np.abs(X[:, -1]), atol=1e-6), "X grid is not symmetric."
    assert np.allclose(np.abs(Y[0, :]), np.abs(Y[-1, :]), atol=1e-6), "Y grid is not symmetric."


@pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
def test_grid_spacing_consistency(plate_resolution):
    # TODO: This should handle non-square plate_sizes
    size = 25.4  # mm
    X, Y = create_grid(size, plate_resolution, indexing="ij")

    # Calculate grid spacings
    dx = np.diff(X[0, :])  # Spacing along x-axis
    dy = np.diff(Y[:, 0])  # Spacing along y-axis

    # Check if all spacings are consistent
    assert np.allclose(dx, dx[0], atol=1e-6), "Grid spacing along X-axis is inconsistent."
    assert np.allclose(dy, dy[0], atol=1e-6), "Grid spacing along Y-axis is inconsistent."


@pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
def test_grid_aspect_ratio(plate_resolution):
    # TODO: This should handle non-square plate_sizes
    size = 25.4  # mm
    X, Y = create_grid(size, plate_resolution, indexing="ij")

    # Expected number of points along one axis
    expected_points = int(np.round(size * plate_resolution))
    assert X.shape == (expected_points, expected_points), (
        f"Grid shape does not match expected resolution. "
        f"Actual: {X.shape}, Expected: ({expected_points}, {expected_points})"
    )
    assert Y.shape == (expected_points, expected_points), (
        f"Grid shape does not match expected resolution. "
        f"Actual: {Y.shape}, Expected: ({expected_points}, {expected_points})"
    )


@pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
def test_grid_center_consistency(plate_resolution):
    # TODO: This should handle non-square plate_sizes
    size = 25.4  # mm
    X, Y = create_grid(size, plate_resolution, indexing="ij")

    # Physical grid center should be 0.0
    c_x, c_y = X.shape[0] // 2, Y.shape[1] // 2
    if len(X) % 2 == 0 and len(Y) % 2 == 0:
        assert np.isclose(X[c_x - 1, c_y - 1], -X[c_x, c_y - 1], atol=1e-6), "X grid center value is incorrect."
        assert np.isclose(Y[c_x - 1, c_y - 1], -Y[c_x - 1, c_y], atol=1e-6), "Y grid center value is incorrect."
    elif len(X) % 2 and len(Y) % 2:
        assert np.isclose(X[c_x, c_y], 0, atol=1e-6), "X grid center value is incorrect."
        assert np.isclose(Y[c_x, c_y], 0, atol=1e-6), "Y grid center value is incorrect."


@pytest.mark.parametrize("plate_resolution", [11.811, 47.244, 94.488])
def test_grid_boundary_symmetry(plate_resolution):
    # TODO: This should handle non-square plate_sizes
    size = 25.4
    X, Y = create_grid(size, plate_resolution, indexing="ij")

    # Check symmetry for X
    assert np.allclose(X[:, 0], X[:, -1], atol=1e-6), "X grid boundary is not symmetric left-to-right."

    # Check symmetry for Y
    assert np.allclose(Y[0, :], Y[-1, :], atol=1e-6), "Y grid boundary is not symmetric top-to-bottom."

    # Assert the shape of X and Y are identical
    assert X.shape == Y.shape, "Grid shapes do not match."


@pytest.mark.parametrize("plate_resolution", [47.244, 94.488])
def test_grid_range(plate_resolution):
    # TODO: This should handle non-square plate_sizes
    size = 25.4  # mm
    X, Y = create_grid(size, plate_resolution)

    print(f"X:\n{X}")
    print(f"Y:\n{Y}")

    # Debug output
    print(f"X range: {np.min(X)} to {np.max(X)}")
    print(f"Y range: {np.min(Y)} to {np.max(Y)}")

    # Check that the grid spans approximately the intended range
    assert np.allclose([np.min(X), np.max(X)], [-size / 2, size / 2], atol=1e-2), (
        f"X grid does not span the correct range. Min: {np.min(X)}, Max: {np.max(X)}"
    )
    assert np.allclose([np.min(Y), np.max(Y)], [-size / 2, size / 2], atol=1e-2), (
        f"Y grid does not span the correct range. Min: {np.min(Y)}, Max: {np.max(Y)}"
    )
