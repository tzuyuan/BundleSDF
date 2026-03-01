# 3D Model Visualization with Texture Support

This project provides comprehensive tools for visualizing 3D models with texture support. It includes interactive viewing, static rendering, mesh analysis, and texture extraction capabilities.

## Features

- **Interactive 3D Viewer**: Real-time 3D model visualization with Open3D
- **Static Rendering**: Generate high-quality images of 3D models using PyRender
- **Mesh Analysis**: Comprehensive analysis of mesh properties and statistics
- **Texture Analysis**: Extract and analyze texture information from models
- **Point Cloud Generation**: Convert meshes to point clouds for analysis
- **Multi-format Export**: Export models in various formats (OBJ, PLY, STL, etc.)

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (optional, for better performance)

### Dependencies

Install the required dependencies:

```bash
pip install -r requirements_visualization.txt
```

Or install manually:

```bash
pip install trimesh open3d pyrender pillow opencv-python matplotlib numpy
```

## Usage

### Command Line Interface

The main visualization script can be used from the command line:

```bash
# Basic analysis and interactive viewer
python visualize_3d_model.py --model your_model.obj

# Interactive 3D viewer only
python visualize_3d_model.py --model your_model.obj --interactive

# Static rendering
python visualize_3d_model.py --model your_model.obj --render --output rendered_image.png

# Comprehensive analysis
python visualize_3d_model.py --model your_model.obj --analyze

# Custom rendering parameters
python visualize_3d_model.py --model your_model.obj --render \
    --width 1024 --height 768 --distance 3.0 --output high_res.png
```

### Command Line Options

- `--model, -m`: Path to the 3D model file (required)
- `--interactive, -i`: Open interactive 3D viewer
- `--render, -r`: Render static image
- `--output, -o`: Output path for rendered image
- `--analyze, -a`: Run comprehensive analysis
- `--width`: Render width (default: 800)
- `--height`: Render height (default: 600)
- `--distance`: Camera distance for rendering (default: 2.0)

### Python API

You can also use the visualization tools programmatically:

```python
from visualize_3d_model import ModelVisualizer

# Create visualizer
visualizer = ModelVisualizer("path/to/model.obj")

# Analyze mesh
visualizer.analyze_mesh()

# Extract texture information
visualizer.extract_texture_info()

# Open interactive viewer
visualizer.visualize_interactive()

# Render static image
visualizer.render_static_image(
    output_path="rendered.png",
    width=1024,
    height=768,
    camera_distance=3.0
)

# Create point cloud
visualizer.create_point_cloud(num_points=10000)

# Export in various formats
visualizer.export_formats(output_dir="exports")

# Run comprehensive analysis
visualizer.run_comprehensive_analysis()
```

## Supported File Formats

### Input Formats
- **OBJ**: Wavefront OBJ files with texture support
- **PLY**: Stanford PLY files
- **STL**: Stereolithography files
- **GLB/GLTF**: glTF binary and text formats
- **FBX**: Autodesk FBX files
- **DAE**: Collada files

### Output Formats
- **OBJ**: Wavefront OBJ
- **PLY**: Stanford PLY
- **STL**: Stereolithography
- **PNG**: Rendered images
- **PLY**: Point clouds

## Texture Support

The visualization tool supports various texture types:

### Vertex Colors
- Per-vertex RGB colors
- Automatic normalization and visualization
- Color distribution analysis

### Face Colors
- Per-face RGB colors
- Statistical analysis of color distribution

### Texture Maps
- UV-mapped texture images
- Automatic texture extraction
- Texture preview generation

## Examples

### Basic Visualization

```python
from visualize_3d_model import ModelVisualizer

# Load and analyze a model
visualizer = ModelVisualizer("model.obj")
visualizer.analyze_mesh()
visualizer.visualize_interactive()
```

### Texture Analysis

```python
# Extract and analyze texture information
visualizer.extract_texture_info()

# This will generate:
# - texture_analysis.png: Color distribution plots
# - texture_image.png: Extracted texture image
# - texture_preview.png: Texture preview
```

### Static Rendering

```python
# Render high-quality image
visualizer.render_static_image(
    output_path="model_render.png",
    width=1920,
    height=1080,
    camera_distance=2.5,
    show_axes=True
)
```

### Point Cloud Generation

```python
# Create point cloud from mesh surface
pcd = visualizer.create_point_cloud(num_points=50000)
# Saves as 'mesh_pointcloud.ply'
```

## Integration with BundleSDF

This visualization tool is designed to work with the BundleSDF project. You can visualize the built 3D models from BundleSDF:

```bash
# Visualize a BundleSDF output model
python visualize_3d_model.py --model path/to/bundlesdf/output/textured_mesh.obj --interactive

# Analyze BundleSDF mesh
python visualize_3d_model.py --model path/to/bundlesdf/output/textured_mesh.obj --analyze
```

## Troubleshooting

### Common Issues

1. **Open3D not available**: Install with `pip install open3d`
2. **PyRender not available**: Install with `pip install pyrender`
3. **Texture not showing**: Check if the model has proper UV coordinates
4. **Performance issues**: Reduce mesh complexity or use point cloud mode

### Dependencies

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install trimesh open3d pyrender pillow opencv-python matplotlib numpy
```

### GPU Support

For better performance with large models:

```bash
# Install CUDA-enabled PyTorch (if using GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Advanced Features

### Custom Rendering

```python
# Custom lighting and materials
scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light)
```

### Mesh Processing

```python
# Clean and process mesh
mesh = visualizer.mesh
mesh.remove_duplicate_vertices()
mesh.remove_degenerate_faces()
mesh.fill_holes()
```

### Texture Extraction

```python
# Extract texture to image file
if visualizer.mesh.visual.kind == 'texture':
    texture_image = visualizer.mesh.visual.material.image
    texture_image.save('extracted_texture.png')
```

## Contributing

To contribute to the visualization tools:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Test with different model formats
5. Submit a pull request

## License

This project is part of the BundleSDF project and follows the same license terms.

## Acknowledgments

- **trimesh**: For mesh loading and processing
- **Open3D**: For interactive 3D visualization
- **PyRender**: For static rendering
- **BundleSDF**: For the underlying 3D reconstruction framework 