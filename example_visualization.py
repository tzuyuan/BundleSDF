#!/usr/bin/env python3
"""
Example script demonstrating 3D model visualization with texture

This script shows how to use the ModelVisualizer class to visualize 3D models
with texture support. It includes examples for different visualization modes.
"""

import os
import sys
import numpy as np

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def example_basic_visualization():
    """Example of basic 3D model visualization."""
    print("=== Basic 3D Model Visualization Example ===")

    # Example model path (you would replace this with your actual model)
    model_path = "path/to/your/model.obj"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please update the model_path variable with a valid 3D model file.")
        return

    try:
        from visualize_3d_model import ModelVisualizer

        # Create visualizer
        visualizer = ModelVisualizer(model_path)

        # Run mesh analysis
        print("\n1. Mesh Analysis:")
        visualizer.analyze_mesh()

        # Extract texture information
        print("\n2. Texture Analysis:")
        visualizer.extract_texture_info()

        # Create point cloud
        print("\n3. Creating Point Cloud:")
        visualizer.create_point_cloud()

        # Export in various formats
        print("\n4. Exporting Model:")
        visualizer.export_formats()

        print("\nBasic visualization completed!")

    except ImportError as e:
        print(f"Import error: {e}")
        print(
            "Please install required dependencies: pip install -r requirements_visualization.txt"
        )
    except Exception as e:
        print(f"Error during visualization: {e}")


def example_interactive_viewer():
    """Example of interactive 3D viewer."""
    print("\n=== Interactive 3D Viewer Example ===")

    model_path = "path/to/your/model.obj"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    try:
        from visualize_3d_model import ModelVisualizer

        visualizer = ModelVisualizer(model_path)

        print("Opening interactive 3D viewer...")
        print("Use mouse to rotate, scroll to zoom, and right-click to pan.")
        print("Close the window when done.")

        visualizer.visualize_interactive()

    except ImportError as e:
        print(f"Import error: {e}")
        print("Open3D is required for interactive visualization.")
    except Exception as e:
        print(f"Error during interactive visualization: {e}")


def example_static_rendering():
    """Example of static image rendering."""
    print("\n=== Static Rendering Example ===")

    model_path = "path/to/your/model.obj"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    try:
        from visualize_3d_model import ModelVisualizer

        visualizer = ModelVisualizer(model_path)

        print("Rendering static image...")

        # Render with different settings
        output_path = "rendered_model.png"
        visualizer.render_static_image(
            output_path=output_path,
            width=1024,
            height=768,
            camera_distance=3.0,
            show_axes=True,
        )

        print(f"Rendered image saved to: {output_path}")

    except ImportError as e:
        print(f"Import error: {e}")
        print("PyRender is required for static rendering.")
    except Exception as e:
        print(f"Error during static rendering: {e}")


def example_comprehensive_analysis():
    """Example of comprehensive model analysis."""
    print("\n=== Comprehensive Analysis Example ===")

    model_path = "path/to/your/model.obj"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    try:
        from visualize_3d_model import ModelVisualizer

        visualizer = ModelVisualizer(model_path)

        print("Running comprehensive analysis...")
        visualizer.run_comprehensive_analysis()

        print("Comprehensive analysis completed!")
        print("Check the generated files:")
        print("- texture_analysis.png: Color distribution analysis")
        print("- texture_image.png: Extracted texture image")
        print("- texture_preview.png: Texture preview")
        print("- mesh_pointcloud.ply: Point cloud representation")
        print("- exports/: Various format exports")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required dependencies.")
    except Exception as e:
        print(f"Error during comprehensive analysis: {e}")


def create_sample_mesh():
    """Create a sample textured mesh for testing."""
    print("\n=== Creating Sample Textured Mesh ===")

    try:
        import trimesh
        import numpy as np

        # Create a simple cube mesh
        vertices = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )

        faces = np.array(
            [
                [0, 1, 2],
                [1, 3, 2],  # Left face
                [4, 6, 5],
                [5, 6, 7],  # Right face
                [0, 4, 1],
                [1, 4, 5],  # Bottom face
                [2, 3, 6],
                [3, 7, 6],  # Top face
                [0, 2, 4],
                [2, 6, 4],  # Back face
                [1, 5, 3],
                [3, 5, 7],  # Front face
            ]
        )

        # Create vertex colors (gradient from red to blue)
        colors = np.zeros((len(vertices), 3), dtype=np.uint8)
        for i, vertex in enumerate(vertices):
            colors[i] = [
                int(255 * vertex[0]),  # Red based on X
                int(255 * vertex[1]),  # Green based on Y
                int(255 * vertex[2]),  # Blue based on Z
            ]

        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Add vertex colors
        mesh.visual.vertex_colors = colors

        # Save the mesh
        output_path = "sample_textured_cube.obj"
        mesh.export(output_path)

        print(f"Sample textured mesh created: {output_path}")
        print("You can now use this file with the visualization script.")

        return output_path

    except ImportError as e:
        print(f"Import error: {e}")
        print("trimesh is required to create sample mesh.")
        return None
    except Exception as e:
        print(f"Error creating sample mesh: {e}")
        return None


def main():
    """Main function demonstrating different visualization features."""
    print("3D Model Visualization Examples")
    print("=" * 40)

    # Create a sample mesh for testing
    sample_mesh_path = create_sample_mesh()

    if sample_mesh_path and os.path.exists(sample_mesh_path):
        print(f"\nUsing sample mesh: {sample_mesh_path}")

        # Update the model path for examples
        import visualize_3d_model

        visualizer = visualize_3d_model.ModelVisualizer(sample_mesh_path)

        # Run examples
        example_basic_visualization()
        example_comprehensive_analysis()

        # Interactive and static rendering examples
        print("\nTo try interactive viewer, run:")
        print(f"python visualize_3d_model.py --model {sample_mesh_path} --interactive")

        print("\nTo try static rendering, run:")
        print(
            f"python visualize_3d_model.py --model {sample_mesh_path} --render --output rendered_sample.png"
        )

    else:
        print("\nNo sample mesh available. Please provide a valid 3D model file.")
        print("\nExample usage:")
        print("python visualize_3d_model.py --model your_model.obj --interactive")
        print("python visualize_3d_model.py --model your_model.obj --analyze")
        print(
            "python visualize_3d_model.py --model your_model.obj --render --output rendered.png"
        )


if __name__ == "__main__":
    main()
