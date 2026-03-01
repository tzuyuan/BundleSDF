import argparse
import glob
import os
import sys
import numpy as np
import trimesh
import imageio.v2 as imageio
import ruamel.yaml

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from Utils import draw_posed_3d_box, draw_xyz_axis

yaml = ruamel.yaml.YAML()


def inverse_SE3(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv


def discover_result_object_dirs(result_dir):
    object_dirs = []
    if os.path.isdir(os.path.join(result_dir, "ob_in_cam")):
        object_dirs.append(result_dir)
    else:
        hit = glob.glob(os.path.join(result_dir, "**", "ob_in_cam"), recursive=True)
        object_dirs.extend(sorted({os.path.dirname(x) for x in hit}))
    return object_dirs


def discover_object_name_from_result_dir(object_result_dir):
    return os.path.basename(object_result_dir.rstrip("/"))


def resolve_mesh_path(mesh_root, object_name):
    name_map = {
        "cracker_box": "003_cracker_box",
        "mustard_bottle": "006_mustard_bottle",
        "extra_large_clamp": "052_extra_large_clamp",
    }
    candidates = [object_name, name_map.get(object_name, object_name)]
    for name in candidates:
        p1 = os.path.join(mesh_root, name, "textured_simple.obj")
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(mesh_root, "models", name, "textured_simple.obj")
        if os.path.exists(p2):
            return p2
    return None


def build_pose_map(pose_dir):
    pose_map = {}
    for p in sorted(glob.glob(os.path.join(pose_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        pose_map[stem] = np.loadtxt(p).reshape(4, 4)
    return pose_map


def build_gt_pose_map(video_dir, object_name):
    gt_dir = os.path.join(video_dir, "annotated_poses", object_name)
    if not os.path.isdir(gt_dir) and object_name == "object_0":
        gt_dir = os.path.join(video_dir, "annotated_poses")
    gt_map = {}
    for p in sorted(glob.glob(os.path.join(gt_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        gt_map[stem] = np.loadtxt(p).reshape(4, 4)
    return gt_map


def choose_output_dir(base_out_dir, object_result_dir, video_name, object_name):
    if base_out_dir is None:
        return os.path.join(object_result_dir, "pose_vis")
    return os.path.join(base_out_dir, video_name, object_name)


def visualize_one_object_result(object_result_dir, mesh_root, out_dir=None, draw_axis=True):
    config_path = os.path.join(object_result_dir, "config_bundletrack.yml")
    if not os.path.exists(config_path):
        print(f"Skip {object_result_dir}: missing {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.load(f)
    video_dir = cfg["data_dir"]
    video_name = os.path.basename(video_dir.rstrip("/"))
    object_name = discover_object_name_from_result_dir(object_result_dir)
    print(f"\nVisualizing {video_name}/{object_name}")

    mesh_file = resolve_mesh_path(mesh_root, object_name)
    if mesh_file is None:
        print(f"Skip {video_name}/{object_name}: mesh not found in {mesh_root}")
        return
    mesh = trimesh.load(mesh_file)
    bmin, bmax = mesh.bounds
    bbox = np.vstack([bmin, bmax])

    color_files = sorted(glob.glob(os.path.join(video_dir, "rgb", "*.png")))
    if len(color_files) == 0:
        print(f"Skip {video_name}/{object_name}: no rgb frames found")
        return
    color_map = {
        os.path.splitext(os.path.basename(p))[0]: p for p in color_files
    }

    K_file = os.path.join(object_result_dir, "cam_K.txt")
    if not os.path.exists(K_file):
        K_file = os.path.join(video_dir, "cam_K.txt")
    K = np.loadtxt(K_file).reshape(3, 3)

    pred_pose_map = build_pose_map(os.path.join(object_result_dir, "ob_in_cam"))
    if len(pred_pose_map) == 0:
        print(f"Skip {video_name}/{object_name}: no predicted poses found")
        return
    gt_pose_map = build_gt_pose_map(video_dir, object_name)

    # Align predictions to GT with the first overlapping frame, matching benchmark logic.
    alignment = np.eye(4)
    overlap_ids = [k for k in sorted(color_map.keys()) if k in pred_pose_map and k in gt_pose_map]
    if len(overlap_ids) > 0:
        k0 = overlap_ids[0]
        alignment = inverse_SE3(pred_pose_map[k0]) @ gt_pose_map[k0]
        print(f"Using frame {k0} for alignment")
    else:
        print("No overlapping pred/gt frame for alignment. Visualizing raw predicted poses.")

    save_dir = choose_output_dir(out_dir, object_result_dir, video_name, object_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to {save_dir}")

    pred_ids = sorted(list(pred_pose_map.keys()))
    for i, frame_id in enumerate(pred_ids):
        if frame_id not in color_map:
            continue
        color = imageio.imread(color_map[frame_id])
        if color.ndim == 3 and color.shape[-1] == 4:
            color = color[..., :3]
        pose = pred_pose_map[frame_id] @ alignment

        vis = draw_posed_3d_box(
            K, color, ob_in_cam=pose, bbox=bbox, line_color=(0, 255, 0), linewidth=2
        )
        if draw_axis:
            vis = draw_xyz_axis(
                color=vis, ob_in_cam=pose, K=K, thickness=3, is_input_rgb=True
            )

        if frame_id in gt_pose_map:
            gt_pose = gt_pose_map[frame_id]
            vis = draw_posed_3d_box(
                K,
                vis,
                ob_in_cam=gt_pose,
                bbox=bbox,
                line_color=(255, 0, 0),
                linewidth=2,
            )
            if draw_axis:
                vis = draw_xyz_axis(
                    color=vis,
                    ob_in_cam=gt_pose,
                    K=K,
                    thickness=3,
                    is_input_rgb=True,
                )

        out_file = os.path.join(save_dir, f"{frame_id}.png")
        imageio.imwrite(out_file, vis)
        if i % 50 == 0:
            print(f"  processed {i}/{len(pred_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help=(
            "Path to an object result dir (<out>/<video>/<object>) OR a parent dir "
            "containing multiple object result dirs."
        ),
    )
    parser.add_argument(
        "--mesh_root",
        type=str,
        default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/YCB_Video_Models/models",
        help="Root containing YCB object meshes.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional output root. Defaults to <object_result_dir>/pose_vis",
    )
    parser.add_argument(
        "--draw_axis",
        type=int,
        default=1,
        help="Set to 0 to disable axis drawing.",
    )
    args = parser.parse_args()

    object_result_dirs = discover_result_object_dirs(args.result_dir)
    if len(object_result_dirs) == 0:
        raise RuntimeError(f"No object result directories found under {args.result_dir}")
    print(f"Found {len(object_result_dirs)} object result directories")
    for d in object_result_dirs:
        visualize_one_object_result(
            object_result_dir=d,
            mesh_root=args.mesh_root,
            out_dir=args.out_dir,
            draw_axis=bool(args.draw_axis),
        )
    print("Done.")
