#!/usr/bin/env python3
"""
3D Model Visualization Script with Texture Support

This script provides multiple visualization options for 3D models with texture:
1. Interactive 3D viewer using Open3D
2. Static rendering using PyRender
3. Mesh analysis and statistics
4. Texture extraction and visualization

Usage:
    python visualize_3d_model.py --model path/to/model.obj
    python visualize_3d_model.py --model path/to/model.obj --interactive
    python visualize_3d_model.py --model path/to/model.obj --render --output rendered_image.png
"""

import os
import sys
import argparse
import numpy as np
import logging

# Try to import optional dependencies
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Some features may be limited.")

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print(
        "Error: trimesh is required but not available. Please install it with: pip install trimesh"
    )
    sys.exit(1)

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. Interactive visualization will be disabled.")

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Image saving features will be disabled.")

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting features will be disabled.")

try:
    import pyrender

    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    print(
        "Warning: PyRender not available. Static rendering features will be disabled."
    )

# Add the current directory to path to import local utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from Utils import *
except ImportError:
    print("Warning: Could not import Utils module. Some features may be limited.")

    # Define basic utility functions if Utils is not available
    def toOpen3dCloud(points, colors=None, normals=None):
        if not OPEN3D_AVAILABLE:
            return None
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None:
            if colors.max() > 1:
                colors = colors / 255.0
            cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        if normals is not None:
            cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        return cloud


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelVisualizer:
    """A comprehensive 3D model visualizer with texture support."""

    def __init__(self, model_path):
        """Initialize the visualizer with a 3D model."""
        self.model_path = model_path
        self.mesh = None
        self.load_model()

    def load_model(self):
        """Load the 3D model from file."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            self.mesh = trimesh.load(self.model_path)

            if isinstance(self.mesh, trimesh.Scene):
                # If it's a scene, get the first mesh
                for name, geometry in self.mesh.geometry.items():
                    if isinstance(geometry, trimesh.Trimesh):
                        self.mesh = geometry
                        break

            if not isinstance(self.mesh, trimesh.Trimesh):
                raise ValueError("Could not load a valid mesh from the file")

            logger.info(f"Model loaded successfully:")
            logger.info(f"  Vertices: {len(self.mesh.vertices)}")
            logger.info(f"  Faces: {len(self.mesh.faces)}")
            logger.info(f"  Has texture: {self.mesh.visual.kind is not None}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def analyze_mesh(self):
        """Analyze and display mesh statistics."""
        if self.mesh is None:
            logger.error("No mesh loaded")
            return

        print("\n" + "=" * 50)
        print("MESH ANALYSIS")
        print("=" * 50)

        # Basic statistics
        print(f"Vertices: {len(self.mesh.vertices):,}")
        print(f"Faces: {len(self.mesh.faces):,}")
        print(f"Edges: {len(self.mesh.edges):,}")

        # Bounding box
        bounds = self.mesh.bounds
        print(f"Bounding box:")
        print(f"  Min: {bounds[0]}")
        print(f"  Max: {bounds[1]}")
        print(f"  Size: {bounds[1] - bounds[0]}")

        # Volume and surface area
        print(f"Surface area: {self.mesh.area:.6f}")
        if self.mesh.is_watertight:
            print(f"Volume: {self.mesh.volume:.6f}")
        else:
            print("Volume: Not available (mesh is not watertight)")

        # Texture information
        if self.mesh.visual.kind is not None:
            print(f"Visual type: {self.mesh.visual.kind}")
            if (
                hasattr(self.mesh.visual, "material")
                and self.mesh.visual.material is not None
            ):
                if hasattr(self.mesh.visual.material, "image"):
                    img = self.mesh.visual.material.image
                    print(f"Texture image size: {img.size}")
                    print(f"Texture image mode: {img.mode}")

        # Mesh properties
        print(f"Watertight: {self.mesh.is_watertight}")
        print(f"Winding consistent: {self.mesh.is_winding_consistent}")
        print(f"Volume: {self.mesh.volume if self.mesh.is_watertight else 'N/A'}")

        # Component analysis
        components = self.mesh.split()
        print(f"Connected components: {len(components)}")
        if len(components) > 1:
            for i, comp in enumerate(components):
                print(
                    f"  Component {i}: {len(comp.vertices)} vertices, {len(comp.faces)} faces"
                )

    def visualize_interactive(self):
        """Open an interactive 3D viewer."""
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D not available. Cannot open interactive viewer.")
            return

        if self.mesh is None:
            logger.error("No mesh loaded")
            return

        logger.info("Opening interactive 3D viewer...")

        # Convert trimesh to Open3D format
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(self.mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(self.mesh.faces)

        # Add normals for better rendering
        o3d_mesh.compute_vertex_normals()

        # Add colors if available
        if self.mesh.visual.kind == "vertex":
            colors = self.mesh.visual.vertex_colors
            if colors is not None:
                # Normalize colors to [0, 1] range
                if colors.max() > 1:
                    colors = colors.astype(np.float32) / 255.0
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window("3D Model Viewer", width=1200, height=800)

        # Add geometry
        vis.add_geometry(o3d_mesh)

        # Set view options
        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().line_width = 1.0

        # Run visualization
        logger.info("Interactive viewer opened. Close the window to continue.")
        vis.run()
        vis.destroy_window()

    def render_static_image(
        self,
        output_path=None,
        width=800,
        height=600,
        camera_distance=2.0,
        show_axes=True,
    ):
        """Render a static image of the 3D model."""
        if not PYRENDER_AVAILABLE:
            logger.error("PyRender not available. Cannot render static image.")
            return None

        if self.mesh is None:
            logger.error("No mesh loaded")
            return None

        logger.info("Rendering static image...")

        # Create scene
        scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[0.1, 0.1, 0.1])

        # Add mesh to scene
        mesh = pyrender.Mesh.from_trimesh(self.mesh)
        scene.add(mesh)

        # Add lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light)

        # Add axes if requested
        if show_axes:
            axes = pyrender.Axes(length=0.5)
            scene.add(axes)

        # Create camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

        # Position camera
        camera_pose = np.eye(4)
        camera_pose[2, 3] = camera_distance  # Move camera back
        scene.add(camera, pose=camera_pose)

        # Create renderer
        renderer = pyrender.OffscreenRenderer(width, height)

        # Render
        color, depth = renderer.render(scene)

        # Convert to PIL Image if available
        if PIL_AVAILABLE:
            image = Image.fromarray(color)

            if output_path:
                image.save(output_path)
                logger.info(f"Rendered image saved to: {output_path}")

            return image
        else:
            logger.warning("PIL not available. Cannot save image.")
            return color

    def extract_texture_info(self):
        """Extract and display texture information."""
        if self.mesh is None:
            logger.error("No mesh loaded")
            return

        print("\n" + "=" * 50)
        print("TEXTURE ANALYSIS")
        print("=" * 50)

        if self.mesh.visual.kind is None:
            print("No texture information available")
            return

        print(f"Visual type: {self.mesh.visual.kind}")

        if self.mesh.visual.kind == "vertex":
            colors = self.mesh.visual.vertex_colors
            if colors is not None:
                print(f"Vertex colors: {colors.shape}")
                print(f"Color range: [{colors.min()}, {colors.max()}]")

                # Show color distribution if matplotlib is available
                if MATPLOTLIB_AVAILABLE:
                    plt.figure(figsize=(12, 4))

                    plt.subplot(1, 3, 1)
                    plt.hist(colors[:, 0], bins=50, alpha=0.7, color="red", label="Red")
                    plt.title("Red Channel Distribution")
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")

                    plt.subplot(1, 3, 2)
                    plt.hist(
                        colors[:, 1], bins=50, alpha=0.7, color="green", label="Green"
                    )
                    plt.title("Green Channel Distribution")
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")

                    plt.subplot(1, 3, 3)
                    plt.hist(
                        colors[:, 2], bins=50, alpha=0.7, color="blue", label="Blue"
                    )
                    plt.title("Blue Channel Distribution")
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")

                    plt.tight_layout()
                    plt.savefig("texture_analysis.png", dpi=150, bbox_inches="tight")
                    plt.close()
                    print("Color distribution plot saved as 'texture_analysis.png'")

        elif self.mesh.visual.kind == "face":
            colors = self.mesh.visual.face_colors
            if colors is not None:
                print(f"Face colors: {colors.shape}")
                print(f"Color range: [{colors.min()}, {colors.max()}]")

        elif self.mesh.visual.kind == "texture":
            if (
                hasattr(self.mesh.visual, "material")
                and self.mesh.visual.material is not None
            ):
                if hasattr(self.mesh.visual.material, "image"):
                    img = self.mesh.visual.material.image
                    print(f"Texture image: {img.size}")
                    print(f"Image mode: {img.mode}")

                    # Save texture image if PIL is available
                    if PIL_AVAILABLE:
                        img.save("texture_image.png")
                        print("Texture image saved as 'texture_image.png'")

                        # Show texture image if matplotlib is available
                        if MATPLOTLIB_AVAILABLE:
                            plt.figure(figsize=(8, 6))
                            plt.imshow(img)
                            plt.title("Texture Image")
                            plt.axis("off")
                            plt.savefig(
                                "texture_preview.png", dpi=150, bbox_inches="tight"
                            )
                            plt.close()
                            print("Texture preview saved as 'texture_preview.png'")

    def create_point_cloud(self, num_points=10000):
        """Create a point cloud from the mesh surface."""
        if self.mesh is None:
            logger.error("No mesh loaded")
            return None

        if not OPEN3D_AVAILABLE:
            logger.error("Open3D not available. Cannot create point cloud.")
            return None

        logger.info("Creating point cloud from mesh surface...")

        # Sample points from the mesh surface
        points, face_indices = self.mesh.sample(num_points, return_index=True)

        # Get colors for the sampled points
        colors = None
        if self.mesh.visual.kind == "vertex":
            # Interpolate vertex colors to face colors
            face_colors = self.mesh.visual.vertex_colors[self.mesh.faces].mean(axis=1)
            colors = face_colors[face_indices]
        elif self.mesh.visual.kind == "face":
            colors = self.mesh.visual.face_colors[face_indices]

        # Create Open3D point cloud
        pcd = toOpen3dCloud(points, colors)

        if pcd is not None:
            # Save point cloud
            o3d.io.write_point_cloud("mesh_pointcloud.ply", pcd)
            logger.info("Point cloud saved as 'mesh_pointcloud.ply'")

        return pcd

    def export_formats(self, output_dir="exports"):
        """Export the model in various formats."""
        if self.mesh is None:
            logger.error("No mesh loaded")
            return

        os.makedirs(output_dir, exist_ok=True)

        logger.info(
            f"Exporting model to various formats in '{output_dir}' directory..."
        )

        # Export as different formats
        formats = {"obj": "model.obj", "ply": "model.ply", "stl": "model.stl"}

        for fmt, filename in formats.items():
            try:
                output_path = os.path.join(output_dir, filename)
                self.mesh.export(output_path)
                logger.info(f"Exported as {fmt}: {output_path}")
            except Exception as e:
                logger.warning(f"Failed to export as {fmt}: {e}")

    def run_comprehensive_analysis(self):
        """Run a comprehensive analysis of the 3D model."""
        logger.info("Starting comprehensive 3D model analysis...")

        # Analyze mesh
        self.analyze_mesh()

        # Extract texture information
        self.extract_texture_info()

        # Create point cloud
        self.create_point_cloud()

        # Export in various formats
        self.export_formats()

        logger.info("Comprehensive analysis completed!")


def main():
    parser = argparse.ArgumentParser(description="3D Model Visualization Tool")
    parser.add_argument(
        "--model", "-m", required=True, help="Path to the 3D model file"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Open interactive 3D viewer"
    )
    parser.add_argument(
        "--render", "-r", action="store_true", help="Render static image"
    )
    parser.add_argument("--output", "-o", help="Output path for rendered image")
    parser.add_argument(
        "--analyze", "-a", action="store_true", help="Run comprehensive analysis"
    )
    parser.add_argument("--width", type=int, default=800, help="Render width")
    parser.add_argument("--height", type=int, default=600, help="Render height")
    parser.add_argument(
        "--distance", type=float, default=2.0, help="Camera distance for rendering"
    )

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return

    try:
        # Create visualizer
        visualizer = ModelVisualizer(args.model)

        # Run requested operations
        if args.analyze:
            visualizer.run_comprehensive_analysis()

        if args.interactive:
            visualizer.visualize_interactive()

        if args.render:
            output_path = args.output or "rendered_model.png"
            visualizer.render_static_image(
                output_path=output_path,
                width=args.width,
                height=args.height,
                camera_distance=args.distance,
            )

        # If no specific action requested, show analysis and interactive viewer
        if not any([args.analyze, args.interactive, args.render]):
            logger.info(
                "No specific action requested. Running analysis and opening interactive viewer..."
            )
            visualizer.analyze_mesh()
            visualizer.visualize_interactive()

    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        return


if __name__ == "__main__":
    main()
