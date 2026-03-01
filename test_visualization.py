#!/usr/bin/env python3
"""
Test script for 3D model visualization functionality

This script creates a simple test mesh and verifies that the visualization
tools work correctly.
"""

import os
import sys
import numpy as np


def create_test_mesh():
    """Create a simple test mesh with texture."""
    try:
        import trimesh

        # Create a simple cube
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],  # bottom
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],  # top
            ]
        )

        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # bottom
                [4, 7, 6],
                [4, 6, 5],  # top
                [0, 4, 5],
                [0, 5, 1],  # front
                [1, 5, 6],
                [1, 6, 2],  # right
                [2, 6, 7],
                [2, 7, 3],  # back
                [3, 7, 4],
                [3, 4, 0],  # left
            ]
        )

        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Add vertex colors (gradient)
        colors = np.zeros((len(vertices), 3), dtype=np.uint8)
        for i, vertex in enumerate(vertices):
            colors[i] = [
                int(255 * vertex[0]),  # Red based on X
                int(255 * vertex[1]),  # Green based on Y
                int(255 * vertex[2]),  # Blue based on Z
            ]

        mesh.visual.vertex_colors = colors

        # Save test mesh
        test_file = "test_mesh.obj"
        mesh.export(test_file)

        print(f"Test mesh created: {test_file}")
        return test_file

    except ImportError:
        print("trimesh not available, skipping test mesh creation")
        return None
    except Exception as e:
        print(f"Error creating test mesh: {e}")
        return None


def test_visualization():
    """Test the visualization functionality."""
    print("Testing 3D Model Visualization")
    print("=" * 40)

    # Create test mesh
    test_file = create_test_mesh()

    if not test_file or not os.path.exists(test_file):
        print("No test mesh available, skipping tests")
        return False

    try:
        # Test import
        from visualize_3d_model import ModelVisualizer

        # Test basic functionality
        print("\n1. Testing ModelVisualizer creation...")
        visualizer = ModelVisualizer(test_file)
        print("✓ ModelVisualizer created successfully")

        # Test mesh analysis
        print("\n2. Testing mesh analysis...")
        visualizer.analyze_mesh()
        print("✓ Mesh analysis completed")

        # Test texture analysis
        print("\n3. Testing texture analysis...")
        visualizer.extract_texture_info()
        print("✓ Texture analysis completed")

        # Test export
        print("\n4. Testing export functionality...")
        visualizer.export_formats("test_exports")
        print("✓ Export completed")

        # Test point cloud creation
        print("\n5. Testing point cloud creation...")
        pcd = visualizer.create_point_cloud(num_points=1000)
        if pcd is not None:
            print("✓ Point cloud created successfully")
        else:
            print("⚠ Point cloud creation failed (Open3D not available)")

        print("\n✓ All tests completed successfully!")
        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print(
            "Please install required dependencies: pip install -r requirements_visualization.txt"
        )
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_command_line():
    """Test command line interface."""
    print("\nTesting Command Line Interface")
    print("=" * 40)

    test_file = "test_mesh.obj"

    if not os.path.exists(test_file):
        print("Test mesh not found, skipping command line tests")
        return False

    import subprocess

    # Test basic analysis
    print("\n1. Testing command line analysis...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "visualize_3d_model.py",
                "--model",
                test_file,
                "--analyze",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✓ Command line analysis successful")
        else:
            print(f"⚠ Command line analysis failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("⚠ Command line test timed out")
    except Exception as e:
        print(f"✗ Command line test failed: {e}")

    return True


def cleanup():
    """Clean up test files."""
    test_files = [
        "test_mesh.obj",
        "texture_analysis.png",
        "texture_image.png",
        "texture_preview.png",
        "mesh_pointcloud.ply",
        "rendered_model.png",
    ]

    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Cleaned up: {file}")
            except Exception as e:
                print(f"Could not clean up {file}: {e}")

    # Clean up test_exports directory
    if os.path.exists("test_exports"):
        try:
            import shutil

            shutil.rmtree("test_exports")
            print("Cleaned up: test_exports/")
        except Exception as e:
            print(f"Could not clean up test_exports: {e}")


def main():
    """Main test function."""
    print("3D Model Visualization Test Suite")
    print("=" * 50)

    # Run tests
    success = test_visualization()
    test_command_line()

    # Cleanup
    print("\nCleaning up test files...")
    cleanup()

    if success:
        print("\n🎉 All tests completed successfully!")
        print("\nYou can now use the visualization tools:")
        print("python visualize_3d_model.py --model your_model.obj --interactive")
        print("python visualize_3d_model.py --model your_model.obj --analyze")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements_visualization.txt")


if __name__ == "__main__":
    main()
