from pathlib import Path

from stl import mesh
import numpy as np


# NOTE: This is meant to be run while in the "stls" directory.
def create_symmetric_object(path: Path = Path("symmetric_object.stl")):
    """
    Create a symmetric octahedron STL with non-shared points for each triangular face.
    """
    # Define the vertices of the octahedron (non-shared points for each face)
    radius = 5.0  # Radius of the octahedron
    top = np.array([0, 0, radius])
    bottom = np.array([0, 0, -radius])
    vertices = [
        # Top faces
        [top, [radius, 0, 0], [0, radius, 0]],  # Top front-right
        [top, [0, radius, 0], [-radius, 0, 0]],  # Top front-left
        [top, [-radius, 0, 0], [0, -radius, 0]],  # Top back-left
        [top, [0, -radius, 0], [radius, 0, 0]],  # Top back-right
        # Bottom faces
        [bottom, [0, radius, 0], [radius, 0, 0]],  # Bottom front-right
        [bottom, [-radius, 0, 0], [0, radius, 0]],  # Bottom front-left
        [bottom, [0, -radius, 0], [-radius, 0, 0]],  # Bottom back-left
        [bottom, [radius, 0, 0], [0, -radius, 0]],  # Bottom back-right
    ]

    # Create face normals and scale to unit vectors
    triangles = []
    normals = []
    for face in vertices:
        triangles.append(face)
        v1, v2, v3 = face
        normal = np.cross(v2 - v1, v3 - v1)
        normals.append(normal / np.linalg.norm(normal))  # Normalize

    # Flatten the list of triangles into the format required by numpy-stl
    data = np.zeros(len(triangles), dtype=mesh.Mesh.dtype)
    for i, (triangle, normal) in enumerate(zip(triangles, normals)):
        data["vectors"][i] = triangle
        data["normals"][i] = normal

    # Write the STL
    non_shared_mesh = mesh.Mesh(data)
    non_shared_mesh.save(str(path))


def fix_normals(path):
    # Reload the mesh
    original_mesh = mesh.Mesh.from_file(path)

    # Recompute and fix normals
    normals = np.cross(original_mesh.v1 - original_mesh.v0, original_mesh.v2 - original_mesh.v0)
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize

    # Ensure outward-facing normals (pointing away from the center)
    triangle_centers = (original_mesh.v0 + original_mesh.v1 + original_mesh.v2) / 3
    outward = np.einsum('ij,ij->i', normals, triangle_centers)  # Dot product with center
    normals[outward < 0] *= -1  # Flip inward-facing normals

    # Update the mesh normals
    original_mesh.normals = normals

    # Save the fixed STL
    original_mesh.save("output_fixed_normals.stl")
    print("Normals fixed and STL saved to 'output_fixed_normals.stl'")

    # Compute triangle centers
    triangle_centers = (original_mesh.v0 + original_mesh.v1 + original_mesh.v2) / 3

    # Recompute normals
    normals = np.cross(original_mesh.v1 - original_mesh.v0, original_mesh.v2 - original_mesh.v0)
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize

    # Correct normals for symmetry
    centroid = np.mean(triangle_centers, axis=0)  # Use mesh centroid
    outward = np.einsum('ij,ij->i', normals, triangle_centers - centroid)  # Dot product
    normals[outward < 0] *= -1  # Flip inward-facing normals

    # Update mesh normals
    original_mesh.normals = normals

    # Save the corrected STL
    original_mesh.save(path)
    print("Normals fixed for symmetry and saved to 'symmetric_object.stl'")


if __name__ == "__main__":
    file_path = Path("symmetric_object.stl")
    create_symmetric_object(file_path)
    fix_normals(file_path)
