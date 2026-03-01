import argparse
import os
import sys
import glob
import numpy as np
import trimesh
import imageio.v2 as imageio
import ruamel.yaml

# Add necessary paths to import Utils and data_reader
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f"{code_dir}/BundleTrack/scripts")

from Utils import draw_posed_3d_box, glcam_in_cvcam, draw_xyz_axis
from data_reader import Ho3dReader

yaml = ruamel.yaml.YAML()


def get_gt_mesh_from_ho3d_root(ho3d_root, video_name):
    video2name = {
        "AP": "019_pitcher_base",
        "MPM": "010_potted_meat_can",
        "SB": "021_bleach_cleanser",
        "SM": "006_mustard_bottle",
    }
    ob_name = None
    for k in video2name:
        if video_name.startswith(k):
            ob_name = video2name[k]
            break

    if ob_name is None:
        return None

    # Check common paths for models
    # Case 1: ho3d_root is .../HO3D_V3
    p1 = os.path.join(ho3d_root, "models", ob_name, "textured_simple.obj")
    if os.path.exists(p1):
        return p1

    # Case 2: ho3d_root is .../HO3D_V3/evaluation (parent has models)
    p2 = os.path.join(
        os.path.dirname(ho3d_root.rstrip("/")), "models", ob_name, "textured_simple.obj"
    )
    if os.path.exists(p2):
        return p2

    return None


def inverse_SE3(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv


def visualize_ho3d(result_dir, out_dir=None, mesh_path=None):
    if not os.path.exists(result_dir):
        print(f"Result directory not found: {result_dir}")
        return

    # Load config to find original data path
    config_path = os.path.join(result_dir, "config_bundletrack.yml")
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.load(f)

    video_dir = config["data_dir"]
    print(f"Loading data from: {video_dir}")

    # Load Reader for input images and camera matrix
    reader = Ho3dReader(video_dir)

    # Load the resulting mesh
    mesh_file_to_load = None

    # 1. Try explicit mesh_path (argument)
    if mesh_path is not None:
        if os.path.isfile(mesh_path):
            mesh_file_to_load = mesh_path
        elif os.path.isdir(mesh_path):
            # Try finding GT mesh in this root
            gt_path = get_gt_mesh_from_ho3d_root(mesh_path, reader.get_video_name())
            if gt_path:
                mesh_file_to_load = gt_path
            else:
                # Try finding result mesh in this dir
                for name in ["textured_mesh.obj", "mesh_real_world.obj"]:
                    p = os.path.join(mesh_path, name)
                    if os.path.exists(p):
                        mesh_file_to_load = p
                        break

    # 2. If not found yet, try result_dir
    if mesh_file_to_load is None:
        if mesh_path is not None:
            print(
                f"Could not find mesh in provided path {mesh_path}, checking result_dir..."
            )

        for name in ["textured_mesh.obj", "mesh_real_world.obj"]:
            p = os.path.join(result_dir, name)
            if os.path.exists(p):
                mesh_file_to_load = p
                break

    if mesh_file_to_load is None or not os.path.exists(mesh_file_to_load):
        print("No mesh found. Cannot compute bounding box.")
        return

    print(f"Loading mesh: {mesh_file_to_load}")
    mesh = trimesh.load(mesh_file_to_load)

    # Compute Bounding Box from mesh vertices (in object frame)
    bmin, bmax = mesh.bounds
    bbox = np.vstack([bmin, bmax])  # (2,3)

    # Gather poses for alignment
    # We need to load ALL estimated poses and GT poses first to align the sequence
    out_poses = []
    gt_poses = []
    gt_ids = []

    print("Loading poses for alignment...")
    for i in range(len(reader.color_files)):
        pose_file = os.path.join(result_dir, "ob_in_cam", f"{reader.id_strs[i]}.txt")
        if os.path.exists(pose_file):
            # Load estimated pose
            # NOTE: BundleTrack saves poses that map from object frame to camera frame
            # However, we need to verify if there's any extra offset.
            # Based on provided reference code, pipeline outputs 'out_pose' directly.
            # The reference code aligns: pred_poses = pred_poses @ inverse_SE3(pred_poses[0]) @ gt_poses[0]

            est_pose = np.loadtxt(pose_file)
            out_poses.append(est_pose)

            gt_pose = reader.get_gt_pose(i)
            if gt_pose is not None:
                gt_ids.append(i)
                # In data_reader.py: get_gt_pose returns glcam_in_cvcam @ ob_in_cam_gt
                # This is already in OpenCV camera frame
                gt_poses.append(gt_pose)
            else:
                # If we have an estimate but no GT, we can't use this frame for alignment check
                # (though we only need the first valid pair for alignment)
                pass
        else:
            # Missing estimate
            out_poses.append(
                np.eye(4)
            )  # Placeholder, won't be used if not in gt_ids or handled carefully

    out_poses = np.array(out_poses)

    # We only care about alignment if we have at least one frame with both Est and GT
    alignment_transform = np.eye(4)
    if len(gt_ids) > 0:
        # Align using the first frame where we have both
        first_idx = gt_ids[0]
        pose_est_0 = out_poses[first_idx]
        pose_gt_0 = gt_poses[0]  # corresponding to first_idx since we appended in order

        # Alignment logic from reference:
        # pred_poses = pred_poses @ inverse_SE3(pred_poses[0]) @ gt_poses[0]
        # This implies: New_Pose_i = Old_Pose_i @ (Old_Pose_0^-1 @ GT_Pose_0)
        # So the alignment transform is (Old_Pose_0^-1 @ GT_Pose_0) applied from the right?
        # Let's check:
        # New_Pose_0 = Old_Pose_0 @ Old_Pose_0^-1 @ GT_Pose_0 = GT_Pose_0. Correct.

        alignment_transform = inverse_SE3(pose_est_0) @ pose_gt_0
        print(f"Computed alignment from frame {first_idx}")
    else:
        print(
            "Warning: No ground truth poses found for alignment. Using raw estimates."
        )

    # Setup output directory
    if out_dir is None:
        out_dir = os.path.join(result_dir, "pose_vis")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving visualizations to {out_dir}")

    # Iterate and draw
    for i, color_file in enumerate(reader.color_files):
        # Load Pose (ob_in_cam)
        pose_file = os.path.join(result_dir, "ob_in_cam", f"{reader.id_strs[i]}.txt")
        if not os.path.exists(pose_file):
            continue

        pose = np.loadtxt(pose_file)

        # Apply alignment: pose = pose @ alignment_transform
        pose = pose @ alignment_transform

        # Load Image
        color = imageio.imread(color_file)

        # Draw estimated bounding box (green)
        # We pass the bbox directly (computed from mesh bounds) and the pose (aligned)
        vis = draw_posed_3d_box(
            reader.K,
            color,
            ob_in_cam=pose,
            bbox=bbox,
            line_color=(0, 255, 0),  # Green
            linewidth=2,
        )
        vis = draw_xyz_axis(
            color=vis, ob_in_cam=pose, K=reader.K, thickness=3, is_input_rgb=True
        )

        # Draw GT bounding box if available (red)
        try:
            # get_gt_pose returns the pose in OpenCV convention already (glcam_in_cvcam applied)
            pose_gt = reader.get_gt_pose(i)

            if pose_gt is not None:
                vis = draw_posed_3d_box(
                    reader.K,
                    vis,
                    ob_in_cam=pose_gt,
                    bbox=bbox,
                    line_color=(255, 0, 0),  # Red
                    linewidth=2,
                )
                vis = draw_xyz_axis(
                    color=vis,
                    ob_in_cam=pose_gt,
                    K=reader.K,
                    thickness=3,
                    is_input_rgb=True,
                )
        except Exception as e:
            # Ignore GT errors, just don't draw it
            # print(e)
            pass

        # Save
        out_file = os.path.join(out_dir, f"{reader.id_strs[i]}.png")
        imageio.imwrite(out_file, vis)

        if i % 50 == 0:
            print(f"Processed {i}/{len(reader.color_files)} frames")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to the output folder of the video (e.g. results/bundlesdf/ho3d/SM1)",
    )
    parser.add_argument(
        "--mesh_root",
        type=str,
        default="/home/justin/data/HO3D_V3/evaluation/",
        help="Optional path to the mesh file or directory containing the mesh. Overrides default search in result_dir.",
    )
    args = parser.parse_args()

    visualize_ho3d(args.result_dir, mesh_path=args.mesh_root)
