# Mesh Viewer for Research

A powerful 3D mesh visualization and analysis tool built with OpenGL and Python. Load, inspect, and analyze 3D mesh files with real-time rendering and comprehensive mesh analysis.

## Features

### Mesh Loading & Formats
- Support for multiple 3D mesh formats: **OBJ**, **STL**, **PLY**, **GLB**, **OFF**
- Automatic mesh normalization and centering
- Scene concatenation for multi-object files

### Rendering Modes
- **Solid Mode**: Fill rendering with lighting
- **Wireframe Mode**: Edge-based visualization  
- **Combined Mode**: Solid + wireframe overlay
- Automatic camera rotation around mesh
- Manual object rotation and scaling controls

### Mesh Analysis & Visualization
- **Intersected Faces Detection**: Automatically detects and highlights self-intersecting triangles
- **Face Normals Display**: Visualize per-face normal vectors (green)
- **Vertex Normals Display**: Visualize per-vertex normal vectors (blue)
- **Point Cloud View**: Sample-based point cloud representation (yellow)

### Mesh Information Display
Displays comprehensive mesh statistics in the console:
- **Statistics**: Vertex/face/edge count, genus, component count
- **Properties**: Watertight, manifold, convex, winding consistency, self-intersection status
- **Analysis**: Area, volume, bounds, center of mass, extents
- **Edge Info**: Internal/boundary edges, connectivity stats, edge lengths, aspect ratio
- **Face Info**: Intersected faces, degenerate faces

### Color Themes
- **Dark Theme**: Dark background with light mesh colors (default)
- **Light Theme**: Light background with dark mesh colors
- Dynamic color adaptation for all visualization elements

### Screenshot Export
- Capture current view to image file
- Auto-crop to remove empty space
- Export formats: **PNG**, **JPEG**, **PDF**
- File dialog for save location selection

## Keyboard Controls

| Key | Action | Description |
|-----|--------|-------------|
| **O** | Open File | Open file dialog to load a mesh file |
| **J** | Solid Mode | Render mesh with filled polygons |
| **K** | Wireframe Mode | Render mesh with edges only |
| **L** | Combined Mode | Render both solid and wireframe |
| **I** | Toggle Intersected Faces | Highlight self-intersecting faces in orange |
| **N** | Toggle Face Normals | Show/hide per-face normal vectors (green) |
| **M** | Toggle Vertex Normals | Show/hide per-vertex normal vectors (blue) |
| **P** | Toggle Point Cloud | Show/hide sampled point cloud (yellow, 8192 points) |
| **U** | Toggle Color Theme | Switch between dark and light theme |
| **C** | Capture Screenshot | Save current view as PNG/JPEG/PDF |
| **SPACE** | Toggle Camera Rotation | Switch between automatic rotation and manual control |
| **R** | Reset Camera | Reset camera to default position and angle |
| **W** | Rotate Object Up | Rotate object upward (X axis) |
| **A** | Rotate Object Left | Rotate object left (Y axis) |
| **S** | Rotate Object Down | Rotate object downward (X axis) |
| **D** | Rotate Object Right | Rotate object right (Y axis) |
| **Q** | Roll Object Left | Roll object left (Z axis) |
| **E** | Roll Object Right | Roll object right (Z axis) |
| **Z** | Scale Down | Scale object down |
| **X** | Scale Up | Scale object up |

## Installation

### Requirements
- Python 3.7+
- OpenGL-capable graphics card

### Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate meshviewer

# Or install dependencies manually
pip install glfw PyOpenGL numpy pyrr pillow trimesh python-fcl colorama
```

## Usage

```bash
python main.py
```

1. Press **O** to open a mesh file
2. Use **J/K/L** to switch between rendering modes
3. Press **I** to highlight any self-intersecting faces
4. Press **N/M** to visualize normals
5. Press **U** to toggle between dark and light themes
6. Press **C** to save a screenshot
7. Check the console for detailed mesh analysis
8. Use **W/A/S/D/Q/E** to rotate the object and **Z/X** to scale

## Technical Details

- **Rendering**: OpenGL 3.3 Core Profile with GLSL shaders
- **Self-Intersection Detection**: FCL (Flexible Collision Library) BVH collision detection
- **Mesh Processing**: Trimesh library for geometry operations
- **Platform**: Cross-platform (Windows, Linux, macOS)

## License

See [LICENSE](LICENSE) file for details.