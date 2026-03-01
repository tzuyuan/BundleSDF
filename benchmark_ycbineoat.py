from Utils import *
import argparse
import pandas as pd

code_dir = os.path.dirname(os.path.realpath(__file__))
from BundleTrack.scripts.data_reader import *


def benchmark_one_video(method, video_dir, out_dir):
    print("\n", video_dir)
    reader = YcbineoatReader(video_dir)
    video_name = reader.get_video_name()

    pred_dir = f"{out_dir}/{video_name}"
    pose_files = sorted(glob.glob(f"{pred_dir}/ob_in_cam/*.txt"))
    if len(pose_files) < len(reader.color_files):
        raise RuntimeError(
            f"Pose file missing for {video_dir}: {len(pose_files)} found"
        )

    pred_poses = []
    for i in range(len(reader.color_files)):
        pred_poses.append(np.loadtxt(pose_files[i]).reshape(4, 4))
    pred_poses = np.asarray(pred_poses)

    gt_poses = []
    ids = []
    for i in range(len(reader.color_files)):
        gt_pose = reader.get_gt_pose(i)
        if gt_pose is None:
            continue
        gt_poses.append(gt_pose)
        ids.append(i)

    if len(ids) == 0:
        raise RuntimeError(f"No GT poses found for {video_dir}")

    ids = np.asarray(ids)
    gt_poses = np.asarray(gt_poses)
    pred_poses = pred_poses[ids]

    # Align using the first valid frame, matching the HO3D benchmark convention.
    pred_poses = pred_poses @ np.linalg.inv(pred_poses[0]) @ gt_poses[0]

    mesh = reader.get_gt_mesh()
    adi_errs = []
    add_errs = []
    for i in range(len(pred_poses)):
        adi_errs.append(adi_err(pred_poses[i], gt_poses[i], mesh.vertices.copy()))
        add_errs.append(add_err(pred_poses[i], gt_poses[i], mesh.vertices.copy()))

    adi_errs = np.asarray(adi_errs)
    add_errs = np.asarray(add_errs)
    adds_auc = compute_auc(adi_errs) * 100
    add_auc = compute_auc(add_errs) * 100

    print(
        f"video {video_name}, ADD-S_err: {adi_errs.mean()*100:.2f}[cm], "
        f"ADD_errs: {add_errs.mean()*100:.2f}[cm], "
        f"ADD-S_AUC: {adds_auc:.2f}, ADD_AUC: {add_auc:.2f}"
    )

    return {
        f"{method}/{video_name}/ADDS(cm)": adi_errs * 100,
        f"{method}/{video_name}/ADD(cm)": add_errs * 100,
        f"{method}/{video_name}/ADDS_AUC(%)": adds_auc,
        f"{method}/{video_name}/ADD_AUC(%)": add_auc,
    }


def discover_video_dirs(video_root):
    # YCBInEOAT videos are folders with rgb/*.png.
    candidate_dirs = sorted(glob.glob(f"{video_root}/*"))
    video_dirs = []
    for d in candidate_dirs:
        if os.path.isdir(d) and len(glob.glob(f"{d}/rgb/*.png")) > 0:
            video_dirs.append(d)
    return video_dirs


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
        default="/home/justin/data/YCBInEOAT",
        help="Root dir used when --video_dirs is empty.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/bundlesdf/ycbineoat",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/bundlesdf/ycbineoat/logs",
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
    skipped_videos = []
    failed_videos = []
    for video_dir in video_dirs:
        try:
            out = benchmark_one_video(method, video_dir, args.out_dir)
            out_data.update(out)
        except Exception as e:
            msg = str(e)
            if "Pose file missing" in msg or "No GT poses found" in msg:
                print(f"Skip {video_dir}: {e}")
                skipped_videos.append(video_dir)
            else:
                print(f"Error benchmarking {video_dir}: {e}")
                failed_videos.append(video_dir)

    print("\n===== Benchmark Summary =====")
    print(f"Succeeded: {len(video_dirs) - len(skipped_videos) - len(failed_videos)}")
    print(f"Skipped: {len(skipped_videos)}")
    print(f"Failed: {len(failed_videos)}")
    if len(skipped_videos) > 0:
        print("Skipped videos:")
        for video_dir in skipped_videos:
            print(f"  - {video_dir}")
    if len(failed_videos) > 0:
        print("Failed videos:")
        for video_dir in failed_videos:
            print(f"  - {video_dir}")

    if len(out_data) == 0:
        raise RuntimeError("No successful videos to benchmark.")

    out = {}
    for k, v in out_data.items():
        metric = k.split("/")[-1]
        if metric not in out:
            out[metric] = []
        if isinstance(v, np.ndarray):
            out[metric].append(v.mean())
        else:
            out[metric].append(v)

    for metric in out:
        print(f"{metric}: {np.asarray(out[metric]).mean():.3f}")

    with open(f"{args.log_dir}/ycbineoat_{method}.pkl", "wb") as ff:
        print("out_data", out_data.keys())
        pickle.dump(out_data, ff)

    video_names = []
    metrics = []
    for k in out_data:
        tmp = k.split("/")
        video_names.append(tmp[1])
        metrics.append(tmp[2])
    video_names = list(np.unique(video_names))
    metrics = list(np.unique(metrics))

    cols = {"videos": video_names}
    for video_name in video_names:
        for metric in metrics:
            if metric not in cols:
                cols[metric] = []
            key = f"{method}/{video_name}/{metric}"
            value = out_data[key]
            if isinstance(value, np.ndarray):
                value = value.mean()
            cols[metric].append(float(value))

    df = pd.DataFrame(cols, index=[method] * len(video_names))

    mean_dict = {}
    for col in cols:
        if col == "videos":
            continue
        mean_dict[col] = df[col].mean()
    df_mean = pd.DataFrame(mean_dict, index=["ALL"])
    df = pd.concat([df, df_mean])

    df.to_excel(f"{args.log_dir}/ycbineoat_{method}.xlsx", sheet_name="per_ob")
