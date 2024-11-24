import numpy as np
import pytest
from stl import mesh


def check_point_symmetry(points: np.ndarray, axis: int, tol: float) -> bool:
    """
    Check if points are symmetric along the specified axis.
    """
    # Reflect points along the specified axis
    reflected = points.copy()
    reflected[:, axis] *= -1  # Negate the specified axis
    # Check if reflected points exist in the original set
    diffs = np.linalg.norm(points[:, None, :] - reflected[None, :, :], axis=2)
    return np.any(diffs < tol, axis=1).all()


def check_normal_symmetry(normals: np.ndarray, points: np.ndarray, axis: int, tol: float) -> bool:
    """
    Check if normals are symmetric along the specified axis.
    """
    reflected_normals = normals.copy()
    reflected_normals[:, axis] *= -1  # Flip normals along the axis
    reflected_points = points.copy()
    reflected_points[:, axis] *= -1

    diffs = np.linalg.norm(normals[:, None, :] - reflected_normals[None, :, :], axis=2)
    return np.any(diffs < tol, axis=1).all()


@pytest.mark.parametrize("stl_file, symmetrical", [("cgh/stls/symmetric_object.stl", True)])
def test_stl_symmetry(stl_file, symmetrical):
    """
    Test the symmetry of the points and normals in an STL file.

    Parameters
    ----------
    stl_file : str
        Path to the STL file.
    """
    loaded_mesh = mesh.Mesh.from_file(str(stl_file))
    points = np.vstack([loaded_mesh.v0, loaded_mesh.v1, loaded_mesh.v2])
    normals = np.array(loaded_mesh.normals)

    assert len(points)
    assert len(normals)

    axis = 0  # Check symmetry along the X-axis
    tol = 1e-6

    is_point_symmetric = check_point_symmetry(points, axis, tol)
    is_normal_symmetric = check_normal_symmetry(normals, points, axis, tol)

    if symmetrical:
        assert is_point_symmetric, f"Points in {stl_file} are not symmetric, but should be."
        assert is_normal_symmetric, f"Normals in {stl_file} are not symmetric, but should be."
    else:
        assert not is_point_symmetric, f"Points in {stl_file} are symmetric, but should not be."
