from cgh.types import Point3D


def generate_test_points_normals(n_points: int = 1) -> tuple[list[Point3D], list[Point3D]]:
    """Generate test points and normals for wave field testing."""
    points = [Point3D(0.0, 0.0, 10.0) for _ in range(n_points)]
    normals = [Point3D(0.0, 0.0, -1.0) for _ in range(n_points)]
    return points, normals
