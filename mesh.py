import trimesh
import numpy as np
import fcl
from colorama import Fore, Style, init

from constants import (
    COPLANAR_TOLERANCE,
    NORMALIZE_BOUND,
    MANIFOLD_EDGE_COUNT,
    FORMAT_LABEL_WIDTH,
    FORMAT_PRECISION_FLOAT,
    FORMAT_PRECISION_COORD
)

# Initialize colorama for Windows compatibility
init(autoreset=True)


class MeshInfo:
    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.intersected_face_ids = get_intersected_tria_ids(mesh)
        self.non_watertight_components = mesh.split(only_watertight=False)
        self.watertight_components = mesh.split(only_watertight=True)
        self.genus = 1 - (len(mesh.vertices) - len(mesh.edges_unique) + len(mesh.faces)) / 2

        self.vertex_defects: np.ndarray = mesh.vertex_defects
        self.vertex_degree: np.ndarray = mesh.vertex_degree

        self.edges_unique: np.ndarray
        self.edges_counts: np.ndarray
        self.edges_unique, self.edges_counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        self.vertex_connectivity = np.bincount(self.edges_unique.flatten(), minlength=len(mesh.vertices))
        self.edges_unique_length: np.ndarray
        self.edges_unique_length = self.mesh.edges_unique_length

        self.nondegenerate_faces_mask = mesh.nondegenerate_faces()
        self.num_degenerate_faces = np.sum(~self.nondegenerate_faces_mask).item()
        self.num_nondegenerate_faces = np.sum(self.nondegenerate_faces_mask).item()
        self.face_angles: np.ndarray = mesh.face_angles
        self.face_areas: np.ndarray = mesh.area_faces
        self.face_adjacency_angles: np.ndarray = mesh.face_adjacency_angles
        
        self.stats = {
            "#vertices": len(mesh.vertices),
            "#faces": len(mesh.faces),
            "#edges": len(mesh.edges_unique),
            "genus": self.genus,
            "#components": mesh.body_count,
            "#components[split][watertight=True]": len(self.watertight_components),
            "#components[split][non_watertight=True]": len(self.non_watertight_components),
        }

        self.properties = {
            "is_watertight": mesh.is_watertight,
            "is_empty": mesh.is_empty,
            "is_winding_consistent": mesh.is_winding_consistent,
            "is_convex": mesh.is_convex,
            "is_manifold[ignore intersection]": is_manifold(mesh),
            "is_manifold": is_manifold(mesh) and len(self.intersected_face_ids) == 0,
            "mutable": mesh.mutable,
            "is_intersecting": len(self.intersected_face_ids) > 0,
        }

        self.analysis = {
            "area": mesh.area,
            "volume": mesh.volume,
            "bounds": mesh.bounds,
            "center_mass": mesh.center_mass,
            "centroid": mesh.centroid,
            "extents": mesh.extents,
        }

        self.vertices_info = {
            "#coplanar_vertices": np.sum(np.abs(self.vertex_defects) < COPLANAR_TOLERANCE).item(),
            "#convex_vertices": np.sum(self.vertex_defects > 0).item(),
            "#concave_vertices": np.sum(self.vertex_defects < 0).item(),
            "min_vertex_degree": int(self.vertex_degree.min()),
            "max_vertex_degree": int(self.vertex_degree.max()),
        }

        self.edges_info = {
            "#internal_edges": np.sum(self.edges_counts == 2).item(),
            "#boundary_edges": np.sum(self.edges_counts == 1).item(),
            "min_connectivity": int(self.vertex_connectivity.min()),
            "max_connectivity": int(self.vertex_connectivity.max()),
            "avg_connectivity": float(self.vertex_connectivity.mean()),
            "min_edge_length[mel]": float(self.edges_unique_length.min()),
            "max_edge_length[mal]": float(self.edges_unique_length.max()),
        }
        self.edges_info["aspect_ratio[ar][mal/mel]"] = self.edges_info["max_edge_length[mal]"] / self.edges_info["min_edge_length[mel]"] if self.edges_info["min_edge_length[mel]"] > 0 else float('inf')

        self.faces_info = {
            "#intersected_faces": len(self.intersected_face_ids),
            "#degenerate_faces": self.num_degenerate_faces,
            "#non_degenerate_faces": self.num_nondegenerate_faces,
            "min_face_angle[rad]": float(self.face_angles.min()),
            "max_face_angle[rad]": float(self.face_angles.max()),
            "min_face_angle[deg]": float(np.degrees(self.face_angles.min())),
            "max_face_angle[deg]": float(np.degrees(self.face_angles.max())),
            "min_face_area": float(self.face_areas.min()),
            "max_face_area": float(self.face_areas.max()),
            "min_dihedral_angle[deg]": float(np.degrees(np.min(self.face_adjacency_angles))),
            "max_dihedral_angle[deg]": float(np.degrees(np.max(self.face_adjacency_angles))),
        }
    
    def __str__(self):
        def format_bool(value):
            """Format boolean values with color."""
            if value is True:
                return f"{Fore.GREEN}True{Style.RESET_ALL}"
            elif value is False:
                return f"{Fore.RED}False{Style.RESET_ALL}"
            return str(value)
        
        def format_value(value):
            """Format values with appropriate color based on type."""
            if isinstance(value, bool):
                return format_bool(value)
            elif isinstance(value, (int, float)):
                return f"{Fore.YELLOW}{value:,}{Style.RESET_ALL}" if isinstance(value, int) else f"{Fore.YELLOW}{value:.{FORMAT_PRECISION_FLOAT}f}{Style.RESET_ALL}"
            else:
                return f"{Fore.WHITE}{value}{Style.RESET_ALL}"
        
        info_str = f"{Fore.CYAN}{Style.BRIGHT}╔═══ Mesh Information ═══╗{Style.RESET_ALL}\n"
        
        # Statistics - group #vertices, #faces, #edges on same row
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Statistics:{Style.RESET_ALL}\n"
        info_str += f"  {Fore.CYAN}{'#vertices / #faces / #edges':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL}"
        info_str += f" {format_value(self.stats['#vertices'])} / {format_value(self.stats['#faces'])} / {format_value(self.stats['#edges'])}\n"
        
        for key, value in self.stats.items():
            if key not in ['#vertices', '#faces', '#edges']:
                formatted_value = format_value(value)
                info_str += f"  {Fore.CYAN}{key:.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {formatted_value}\n"
        
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Properties:{Style.RESET_ALL}\n"
        # Row 1: Topological properties
        info_str += f"  {Fore.CYAN}watertight:{Style.RESET_ALL} {format_bool(self.properties['is_watertight'])}  "
        info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}manifold (ignore intersection):{Style.RESET_ALL} {format_bool(self.properties['is_manifold[ignore intersection]'])}  {Fore.CYAN}manifold:{Style.RESET_ALL} {format_bool(self.properties['is_manifold'])}  "
        info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}winding_consistent:{Style.RESET_ALL} {format_bool(self.properties['is_winding_consistent'])}\n"
        
        # Row 2: Geometric and state properties
        info_str += f"  {Fore.CYAN}convex:{Style.RESET_ALL} {format_bool(self.properties['is_convex'])}  "
        info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}empty:{Style.RESET_ALL} {format_bool(self.properties['is_empty'])}  "
        info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}intersecting:{Style.RESET_ALL} {format_bool(self.properties['is_intersecting'])}  "
        info_str += f"{Fore.WHITE}|{Style.RESET_ALL} {Fore.CYAN}mutable:{Style.RESET_ALL} {format_bool(self.properties['mutable'])}\n"
        
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Analysis:{Style.RESET_ALL}\n"
        for key, value in self.analysis.items():
            if key == "bounds":
                value_str = f"{Fore.YELLOW}[[{value[0][0]:.{FORMAT_PRECISION_COORD}f}, {value[0][1]:.{FORMAT_PRECISION_COORD}f}, {value[0][2]:.{FORMAT_PRECISION_COORD}f}], "
                value_str += f"[{value[1][0]:.{FORMAT_PRECISION_COORD}f}, {value[1][1]:.{FORMAT_PRECISION_COORD}f}, {value[1][2]:.{FORMAT_PRECISION_COORD}f}]]{Style.RESET_ALL}"
                info_str += f"  {Fore.CYAN}{key:.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {value_str}\n"
            elif key == "extents":
                value_str = f"{Fore.YELLOW}[l = {value[0]:.{FORMAT_PRECISION_COORD}f}, w = {value[1]:.{FORMAT_PRECISION_COORD}f}, h = {value[2]:.{FORMAT_PRECISION_COORD}f}]{Style.RESET_ALL}"
                info_str += f"  {Fore.CYAN}{key:.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {value_str}\n"
            else:
                formatted_value = format_value(value)
                info_str += f"  {Fore.CYAN}{key:.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {formatted_value}\n"
        
        # Vertices Info - group related items
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Vertices Info:{Style.RESET_ALL}\n"
        info_str += f"  {Fore.CYAN}{'#coplanar / #convex / #concave':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.vertices_info['#coplanar_vertices'])} / "
        info_str += f"{format_value(self.vertices_info['#convex_vertices'])} / "
        info_str += f"{format_value(self.vertices_info['#concave_vertices'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max vertex_degree':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.vertices_info['min_vertex_degree'])} / "
        info_str += f"{format_value(self.vertices_info['max_vertex_degree'])}\n"
        
        # Edges Info - group min/max pairs
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Edges Info:{Style.RESET_ALL}\n"
        info_str += f"  {Fore.CYAN}{'#internal / #boundary edges':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.edges_info['#internal_edges'])} / "
        info_str += f"{format_value(self.edges_info['#boundary_edges'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max / avg connectivity':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.edges_info['min_connectivity'])} / "
        info_str += f"{format_value(self.edges_info['max_connectivity'])} / "
        info_str += f"{format_value(self.edges_info['avg_connectivity'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max edge_length[mel/mal]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.edges_info['min_edge_length[mel]'])} / "
        info_str += f"{format_value(self.edges_info['max_edge_length[mal]'])}\n"
        
        info_str += f"  {Fore.CYAN}{'aspect_ratio[ar][mal/mel]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.edges_info['aspect_ratio[ar][mal/mel]'])}\n"

        # Faces Info - group min/max pairs
        info_str += f"\n{Fore.MAGENTA}{Style.BRIGHT}Faces Info:{Style.RESET_ALL}\n"
        info_str += f"  {Fore.CYAN}{'#intersected_faces':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} {format_value(self.faces_info['#intersected_faces'])}\n"
        info_str += f"  {Fore.CYAN}{'#degenerate / #non_degenerate':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['#degenerate_faces'])} / "
        info_str += f"{format_value(self.faces_info['#non_degenerate_faces'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max face_angle[rad]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['min_face_angle[rad]'])} / "
        info_str += f"{format_value(self.faces_info['max_face_angle[rad]'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max face_angle[deg]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['min_face_angle[deg]'])} / "
        info_str += f"{format_value(self.faces_info['max_face_angle[deg]'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max face_area':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['min_face_area'])} / "
        info_str += f"{format_value(self.faces_info['max_face_area'])}\n"
        
        info_str += f"  {Fore.CYAN}{'min / max dihedral_angle[deg]':.<{FORMAT_LABEL_WIDTH}}{Style.RESET_ALL} "
        info_str += f"{format_value(self.faces_info['min_dihedral_angle[deg]'])} / "
        info_str += f"{format_value(self.faces_info['max_dihedral_angle[deg]'])}\n"
        
        info_str += f"\n{Fore.CYAN}{Style.BRIGHT}╚═══════════════════════╝{Style.RESET_ALL}"
        return info_str

def is_manifold(mesh: trimesh.Trimesh) -> bool:
    """
    Check if a mesh is manifold.

    Parameters:
    mesh (trimesh.Trimesh): The input mesh.
    Returns:
    bool: True if the mesh is manifold, False otherwise.
    """
    
    edges_sorted = np.sort(mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    is_manifold = np.all(counts == MANIFOLD_EDGE_COUNT).item()

    return is_manifold


def normalize_vertices(vertices: np.ndarray, bound=NORMALIZE_BOUND) -> np.ndarray:
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
    mesh: trimesh.Trimesh = trimesh.load(file_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = mesh.vertices
    faces = mesh.faces

    vertices = normalize_vertices(vertices)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    return mesh

def get_intersected_tria_ids(mesh: trimesh.Trimesh):
    """
    Identify intersected triangle IDs in a mesh using FCL.

    Parameters:
    mesh (trimesh.Trimesh): The input mesh.
    Returns:
    list: A list of intersected triangle IDs.
    """
    # 1. Build the FCL Model
    model = fcl.BVHModel()
    model.beginModel(len(mesh.vertices), len(mesh.faces))
    model.addSubModel(mesh.vertices, mesh.faces)
    model.endModel()

    mesh_obj = fcl.CollisionObject(model, fcl.Transform())

    # 2. Collision Request
    request = fcl.CollisionRequest(enable_contact=True, num_max_contacts=len(mesh.faces) ** 2)
    result = fcl.CollisionResult()
    fcl.collide(mesh_obj, mesh_obj, request, result)

    intersected_ids = set()

    # 3. The "Zero Shared Vertices" Filter
    for contact in result.contacts:
        id1, id2 = contact.b1, contact.b2

        if id1 == id2:
            continue

        # Get the vertex indices for both triangles
        v1 = set(mesh.faces[id1])
        v2 = set(mesh.faces[id2])

        # INTERSECTION LOGIC:
        # If they share 1 or more vertices, they are "touching" (neighbors).
        # We only care if they share 0 vertices AND FCL says they collide.
        if len(v1.intersection(v2)) == 0:
            intersected_ids.add(id1)
            intersected_ids.add(id2)

    return list(intersected_ids)