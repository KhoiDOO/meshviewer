import trimesh
import numpy as np

def normalize_vertices(vertices: np.ndarray, bound=0.95) -> np.ndarray:
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    ori_center = (vmax + vmin) / 2
    ori_scale = 2 * bound / np.max(vmax - vmin)
    vertices = (vertices - ori_center) * ori_scale
    return vertices

def load_mesh(file_path):
    """
    Load a 3D mesh from a file.

    Parameters:
    file_path (str): The path to the mesh file.
    Returns:
    trimesh.Trimesh: The loaded mesh object.
    """
    mesh: trimesh.Trimesh = trimesh.load(file_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = mesh.vertices
    faces = mesh.faces

    vertices = normalize_vertices(vertices)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    return mesh