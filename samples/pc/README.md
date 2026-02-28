# Sample Point Clouds

This folder contains sample point cloud files for testing and demonstrating the point cloud viewer (`main_pc.py`).

## Supported Formats

### `.xyz` Format (Plain Text)
- Simple text format with one point per line
- Format: `X Y Z` (space or tab-separated)
- Optional attributes: `X Y Z R G B` for color data
- Easy to create and parse
- Recommended for testing and custom data

Example:
```
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
```

With color:
```
0.0 0.0 0.0 255 0 0
1.0 0.0 0.0 0 255 0
0.0 1.0 0.0 0 0 255
```

### `.las` Format (LAS/LAZ)
- Industry standard for LiDAR point cloud data
- Binary format with compression support
- Rich metadata (intensity, classification, etc.)
- Efficient for large datasets
- Requires `laspy` library for reading

### `.ply` Format (PLY)
- Flexible polygon file format
- Supports both point cloud and mesh data
- Text or binary encoding
- Column headers specify data attributes

## Test Files

### `airplane.xyz`
**Type**: Simple point cloud in XYZ format
**Description**: Sample 3D point cloud representing an airplane geometry
**Characteristics**:
- Single-precision floating-point coordinates
- No color information (default white)
- Suitable for testing basic point cloud loading and rendering

**Visualization Tips**:
1. Run `python main_pc.py`
2. Press `O` to open the file dialog
3. Select `airplane.xyz`
4. Use mouse controls to rotate and zoom
5. Press `P` for different point sizes
6. Toggle color mapping if color data is available

## Loading Point Clouds

```bash
# Load a single point cloud
python main_pc.py

# The viewer will open, then use the O key to load files
# Or pass the file path as an argument (if supported)
```

## Multi-Cloud Comparison

Load multiple point clouds simultaneously:
1. Press `O` to open file dialog
2. Hold `Cmd` (macOS) or `Ctrl` (Windows/Linux)
3. Select multiple `.xyz`, `.las`, or `.ply` files
4. Clouds will be arranged in a grid layout

## keyboard Controls in main_pc.py

| Key | Action |
|-----|--------|
| **O** | Open point cloud file(s) |
| **J** | Solid rendering (points) |
| **P** | Cycle point sizes |
| **U** | Toggle color theme |
| **C** | Capture screenshot |
| **SPACE** | Toggle camera rotation |
| **R** | Reset camera |
| **W/A/S/D** | Rotate view |
| **Z/X** | Scale points |
| **↑/↓** | Move camera height |

## Creating Your Own Point Cloud Files

### From XYZ Text
Create a file with `.xyz` extension:
```
# points.xyz
10.5 20.3 15.2
11.2 21.1 15.8
12.0 19.5 14.9
```

### From NumPy (Python)
```python
import numpy as np

# Generate random point cloud
points = np.random.rand(10000, 3) * 100

# Save to XYZ format
np.savetxt('pointcloud.xyz', points, fmt='%.6f')

# With color
colors = np.random.randint(0, 256, (10000, 3))
data = np.hstack([points, colors])
np.savetxt('pointcloud_colored.xyz', data, fmt='%d %d %d %.6f %.6f %.6f')
```

### From LiDAR Scans
Use specialized tools like CloudCompare or open3d:
```python
import open3d as o3d

# Load LAS file
pcd = o3d.io.read_point_cloud("scan.las")

# Export to XYZ
o3d.io.write_point_cloud("scan.xyz", pcd)
```

## Technical Notes

- Point clouds are automatically normalized to fit in viewing volume
- Large point clouds (>10M points) may require GPU with sufficient VRAM
- Color data is optional; points default to white when not provided
- RGB values should be in range [0, 255] (will be normalized to [0, 1])
- Coordinate precision is preserved in XYZ format (floating-point)

## Troubleshooting

**File won't load:**
- Check file format matches expected structure
- Ensure coordinates are numeric (not text)
- Verify file encoding is UTF-8

**Points appear as artifacts:**
- Check coordinate ranges (may need normalization)
- Verify point format matches selected file type
- Try with a smaller point cloud first

**Performance issues with large clouds:**
- Reduce point count using external tools (CloudCompare, open3d)
- Try LAS with compression for faster loading
- Use dedicated point cloud processing libraries for preprocessing
