from pathlib import Path

import numpy as np
import pytest
from stl import mesh

from cgh.types import HologramParameters


FLOAT_TYPES = [
    getattr(np, f"float{t}") for t in [32, 64, 128, 256]
    if hasattr(np, f"float{t}")
]


COMPLEX_TYPES = [
    getattr(np, f"complex{t}") for t in [64, 128, 256, 512]
    if hasattr(np, f"complex{t}")
]


def create_test_stl(path: Path = Path("test_object.stl")) -> Path:
    """Create a simple test STL file."""
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3]
    ])

    data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
    for i, face in enumerate(faces):
        data['vectors'][i] = vertices[face]

    test_mesh = mesh.Mesh(data)
    test_mesh.save(path)

    return path


@pytest.fixture
def test_parameters() -> HologramParameters:
    """Standard test parameters."""
    return HologramParameters(
        wavelength=0.532,         # 532nm in mm
        plate_size=25.4,          # 1 inch plate
        plate_resolution=47.244,  # 1200 dpi
        light_source_distance=500.0,
        object_distance=50.0,
        scale_factor=1.0,
        subdivision_factor=1
    )


@pytest.fixture
def test_stl_file(tmp_path: Path) -> Path:
    """Fixture providing a test STL file."""
    return create_test_stl(tmp_path / "test_object.stl")
