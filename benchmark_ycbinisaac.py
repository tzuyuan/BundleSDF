from Utils import *
import argparse
import os
import pandas as pd


def discover_video_dirs(video_root):
    candidate_dirs = sorted(glob.glob(f"{video_root}/*"))
    video_dirs = []
    for d in candidate_dirs:
        if os.path.isdir(d) and len(glob.glob(f"{d}/rgb/*.png")) > 0:
            video_dirs.append(d)
    return video_dirs


def discover_object_names(video_dir):
    names = set()
    masks_root = os.path.join(video_dir, "masks")
    poses_root = os.path.join(video_dir, "annotated_poses")
    if os.path.isdir(masks_root):
        for d in os.listdir(masks_root):
            if os.path.isdir(os.path.join(masks_root, d)):
                names.add(d)
    if os.path.isdir(poses_root):
        for d in os.listdir(poses_root):
            if os.path.isdir(os.path.join(poses_root, d)):
                names.add(d)
    if len(names) == 0:
        return ["object_0"]
    return sorted(list(names))


def build_file_map(files):
    out = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        out[stem] = f
    return out


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


def load_is_obj_in_image_labels(video_dir, obj_name, num_frames):
    label_file = os.path.join(
        video_dir, "is_obj_in_image_labels", obj_name, "is_obj_in_image.npy"
    )
    if not os.path.exists(label_file) and obj_name == "object_0":
        fallback = os.path.join(
            video_dir, "is_obj_in_image_labels", "is_obj_in_image.npy"
        )
        if os.path.exists(fallback):
            label_file = fallback

    if not os.path.exists(label_file):
        raise RuntimeError(f"Missing is_obj_in_image labels: {label_file}")

    labels = np.load(label_file).reshape(-1)
    if len(labels) != num_frames:
        raise RuntimeError(
            f"Label length mismatch for {video_dir}/{obj_name}: "
            f"len(labels)={len(labels)} vs num_frames={num_frames}"
        )
    return labels.astype(bool)


def benchmark_one_object(method, video_dir, obj_name, out_dir, mesh_root):
    print(f"\n{video_dir} / {obj_name}")
    video_name = os.path.basename(video_dir.rstrip("/"))
    pred_dir = os.path.join(out_dir, video_name, obj_name)
    pose_files = sorted(glob.glob(f"{pred_dir}/ob_in_cam/*.txt"))
    if len(pose_files) == 0:
        raise RuntimeError(f"No predicted poses found: {pred_dir}/ob_in_cam")

    color_files = sorted(glob.glob(f"{video_dir}/rgb/*.png"))
    if len(color_files) == 0:
        raise RuntimeError(f"No rgb frames found in {video_dir}")
    id_strs = [os.path.splitext(os.path.basename(x))[0] for x in color_files]
    is_obj_in_image = load_is_obj_in_image_labels(video_dir, obj_name, len(id_strs))
    eval_ids = [id_strs[i] for i in range(len(id_strs)) if is_obj_in_image[i]]
    if len(eval_ids) == 0:
        raise RuntimeError(
            f"No in-image frames from labels for {video_name}/{obj_name}"
        )

    pred_pose_map = {
        os.path.splitext(os.path.basename(p))[0]: np.loadtxt(p).reshape(4, 4)
        for p in pose_files
    }

    gt_pose_dir = os.path.join(video_dir, "annotated_poses", obj_name)
    if not os.path.isdir(gt_pose_dir) and obj_name == "object_0":
        gt_pose_dir = os.path.join(video_dir, "annotated_poses")
    gt_pose_files = sorted(glob.glob(f"{gt_pose_dir}/*.txt"))
    if len(gt_pose_files) == 0:
        raise RuntimeError(f"No GT pose files found for {video_name}/{obj_name}")
    gt_pose_map = {
        os.path.splitext(os.path.basename(p))[0]: np.loadtxt(p).reshape(4, 4)
        for p in gt_pose_files
    }

    missing_pred = [k for k in eval_ids if k not in pred_pose_map]
    if len(missing_pred) > 0:
        raise RuntimeError(
            f"Missing predicted poses for in-image frames ({video_name}/{obj_name}), "
            f"e.g. {missing_pred[:5]}"
        )
    missing_gt = [k for k in eval_ids if k not in gt_pose_map]
    if len(missing_gt) > 0:
        raise RuntimeError(
            f"Missing GT poses for in-image frames ({video_name}/{obj_name}), "
            f"e.g. {missing_gt[:5]}"
        )

    valid_ids = eval_ids
    if len(valid_ids) == 0:
        raise RuntimeError(f"No overlapping pred/gt frames for {video_name}/{obj_name}")

    pred_poses = np.asarray([pred_pose_map[k] for k in valid_ids])
    gt_poses = np.asarray([gt_pose_map[k] for k in valid_ids])

    # Align by first valid frame to match other benchmark scripts.
    pred_poses = pred_poses @ np.linalg.inv(pred_poses[0]) @ gt_poses[0]

    mesh_file = resolve_mesh_path(mesh_root, obj_name)
    if mesh_file is None:
        raise RuntimeError(f"Mesh not found for object {obj_name} in {mesh_root}")
    mesh = trimesh.load(mesh_file)

    adi_errs = []
    add_errs = []
    for i in range(len(pred_poses)):
        verts = mesh.vertices.copy()
        adi_errs.append(adi_err(pred_poses[i], gt_poses[i], verts))
        add_errs.append(add_err(pred_poses[i], gt_poses[i], verts))
    adi_errs = np.asarray(adi_errs)
    add_errs = np.asarray(add_errs)
    adds_auc = compute_auc(adi_errs) * 100
    add_auc = compute_auc(add_errs) * 100

    print(
        f"{video_name}/{obj_name}, ADD-S_err: {adi_errs.mean()*100:.2f}[cm], "
        f"ADD_err: {add_errs.mean()*100:.2f}[cm], "
        f"ADD-S_AUC: {adds_auc:.2f}, ADD_AUC: {add_auc:.2f}"
    )

    key_prefix = f"{method}/{video_name}/{obj_name}"
    return {
        f"{key_prefix}/ADDS(cm)": adi_errs * 100,
        f"{key_prefix}/ADD(cm)": add_errs * 100,
        f"{key_prefix}/ADDS_AUC(%)": adds_auc,
        f"{key_prefix}/ADD_AUC(%)": add_auc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dirs",
        type=str,
        default="",
        help="Comma-separated list of video dirs. Empty means discover all under --video_root.",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="/home/justin/data/test",
        help="Root dir used when --video_dirs is empty.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/bundlesdf/ycbinisaac",
    )
    parser.add_argument(
        "--mesh_root",
        type=str,
        default="/home/justin/data/HO3D_V3/models",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/bundlesdf/ycbinisaac/logs",
    )
    parser.add_argument(
        "--max_objects_per_sequence",
        type=int,
        default=2,
        help="Max objects to evaluate per sequence. Set <=0 for all discovered objects.",
    )
    args = parser.parse_args()

    method = "ours"
    os.makedirs(args.log_dir, exist_ok=True)

    if args.video_dirs.strip():
        video_dirs = [x.strip() for x in args.video_dirs.split(",") if x.strip()]
    else:
        video_dirs = discover_video_dirs(args.video_root)

    if len(video_dirs) == 0:
        raise RuntimeError("No video directories found to benchmark.")

    print(f"Found {len(video_dirs)} videos")
    out_data = {}
    skipped_items = []
    failed_items = []
    for video_dir in video_dirs:
        object_names = discover_object_names(video_dir)
        if args.max_objects_per_sequence > 0:
            object_names = object_names[: args.max_objects_per_sequence]
        for obj_name in object_names:
            item = f"{os.path.basename(video_dir)}/{obj_name}"
            try:
                out = benchmark_one_object(
                    method=method,
                    video_dir=video_dir,
                    obj_name=obj_name,
                    out_dir=args.out_dir,
                    mesh_root=args.mesh_root,
                )
                out_data.update(out)
            except Exception as e:
                msg = str(e)
                if "No predicted poses found" in msg or "No GT pose files found" in msg:
                    print(f"Skip {item}: {e}")
                    skipped_items.append(item)
                else:
                    print(f"Error benchmarking {item}: {e}")
                    failed_items.append(item)

    print("\n===== Benchmark Summary =====")
    total_items = sum(
        [
            (
                len(discover_object_names(v))
                if args.max_objects_per_sequence <= 0
                else min(len(discover_object_names(v)), args.max_objects_per_sequence)
            )
            for v in video_dirs
        ]
    )
    print(f"Succeeded: {len(out_data) // 4}")
    print(f"Skipped: {len(skipped_items)}")
    print(f"Failed: {len(failed_items)}")
    print(f"Total attempted: {total_items}")
    if len(skipped_items) > 0:
        print("Skipped items:")
        for item in skipped_items:
            print(f"  - {item}")
    if len(failed_items) > 0:
        print("Failed items:")
        for item in failed_items:
            print(f"  - {item}")

    if len(out_data) == 0:
        raise RuntimeError("No successful sequence/object pairs to benchmark.")

    out = {}
    for k, v in out_data.items():
        metric = k.split("/")[-1]
        if metric not in out:
            out[metric] = []
        out[metric].append(v.mean() if isinstance(v, np.ndarray) else v)
    for metric in out:
        print(f"{metric}: {np.asarray(out[metric]).mean():.3f}")

    with open(f"{args.log_dir}/ycbinisaac_{method}.pkl", "wb") as ff:
        pickle.dump(out_data, ff)

    keys = sorted(out_data.keys())
    item_names = sorted(list({"/".join(k.split("/")[1:3]) for k in keys}))
    metrics = sorted(list({k.split("/")[-1] for k in keys}))
    cols = {"video_object": item_names}
    for metric in metrics:
        cols[metric] = []
        for item_name in item_names:
            key = f"{method}/{item_name}/{metric}"
            val = out_data.get(key, np.nan)
            if isinstance(val, np.ndarray):
                val = val.mean()
            cols[metric].append(float(val))
    df = pd.DataFrame(cols)
    mean_dict = {metric: df[metric].mean() for metric in metrics}
    mean_dict["video_object"] = "ALL"
    df_mean = pd.DataFrame([mean_dict])
    df = pd.concat([df, df_mean], ignore_index=True)
    df.to_excel(f"{args.log_dir}/ycbinisaac_{method}.xlsx", sheet_name="per_ob")
