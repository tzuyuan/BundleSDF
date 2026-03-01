from bundlesdf import *
import argparse
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
from BundleTrack.scripts.data_reader import *
from segmentation_utils import Segmenter

vid_to_skip = ["YCB_models_with_ply"]


def discover_video_dirs(video_root):
    video_dirs = []
    for video_dir in sorted(glob.glob(f"{video_root}/*")):
        if not os.path.isdir(video_dir):
            continue
        if len(glob.glob(f"{video_dir}/rgb/*.png")) == 0:
            continue
        video_dirs.append(video_dir)
    return video_dirs


def run_one_video(video_dir, out_dir):
    set_seed(0)

    reader = YcbineoatReader(video_dir=video_dir, shorter_side=args.shorter_side)
    video_name = reader.get_video_name()
    if video_name in vid_to_skip:
        return "skipped"
    out_folder = f"{out_dir}/{video_name}/"  # NOTE there has to be a / in the end

    # if os.path.exists(f"{out_folder}/ob_in_cam"):
    #     pose_files = sorted(glob.glob(f"{out_folder}/ob_in_cam/*.txt"))
    #     expected = len(range(0, len(reader.color_files), args.stride))
    #     if len(pose_files) == expected:
    #         print(f"{out_folder} done before, skip")
    #         return "skipped"

    os.system(f"rm -rf {out_folder} && mkdir -p {out_folder}")

    cfg_bundletrack = yaml.load(open(args.track_cfg, "r"))
    cfg_bundletrack["data_dir"] = video_dir
    cfg_bundletrack["SPDLOG"] = int(args.debug_level)
    # cfg_bundletrack["depth_processing"]["percentile"] = 95
    cfg_bundletrack["erode_mask"] = 0
    cfg_bundletrack["debug_dir"] = out_folder
    # cfg_bundletrack["bundle"]["max_BA_frames"] = 10
    # cfg_bundletrack["bundle"]["max_optimized_feature_loss"] = 0.03
    # cfg_bundletrack["feature_corres"]["max_dist_neighbor"] = 0.02
    # cfg_bundletrack["feature_corres"]["max_normal_neighbor"] = 30
    # cfg_bundletrack["feature_corres"]["max_dist_no_neighbor"] = 0.01
    # cfg_bundletrack["feature_corres"]["max_normal_no_neighbor"] = 20
    # cfg_bundletrack["feature_corres"]["map_points"] = True
    # cfg_bundletrack["feature_corres"]["resize"] = 400
    # cfg_bundletrack["feature_corres"]["rematch_after_nerf"] = False
    # cfg_bundletrack["keyframe"]["min_rot"] = 5
    # cfg_bundletrack["ransac"]["inlier_dist"] = 0.01
    # cfg_bundletrack["ransac"]["inlier_normal_angle"] = 20
    # cfg_bundletrack["ransac"]["max_trans_neighbor"] = 0.02
    # cfg_bundletrack["ransac"]["max_rot_deg_neighbor"] = 30
    # cfg_bundletrack["ransac"]["max_trans_no_neighbor"] = 0.01
    # cfg_bundletrack["ransac"]["max_rot_no_neighbor"] = 10
    # cfg_bundletrack["p2p"]["max_dist"] = 0.02
    # cfg_bundletrack["p2p"]["max_normal_angle"] = 45
    cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
    yaml.dump(cfg_bundletrack, open(cfg_track_dir, "w"))

    cfg_nerf = yaml.load(open(f"{SCRIPT_DIR}/config.yml", "r"))
    cfg_nerf["continual"] = True
    cfg_nerf["trunc_start"] = 0.01
    cfg_nerf["trunc"] = 0.01
    cfg_nerf["mesh_resolution"] = 0.005
    cfg_nerf["down_scale_ratio"] = 1
    cfg_nerf["fs_sdf"] = 0.1
    cfg_nerf["far"] = cfg_bundletrack["depth_processing"]["zfar"]
    cfg_nerf["datadir"] = f"{out_folder}/nerf_with_bundletrack_online"
    cfg_nerf["notes"] = ""
    cfg_nerf["expname"] = "nerf_with_bundletrack_online"
    cfg_nerf["save_dir"] = cfg_nerf["datadir"]
    cfg_nerf_dir = f"{out_folder}/config_nerf.yml"
    yaml.dump(cfg_nerf, open(cfg_nerf_dir, "w"))

    if args.use_segmenter:
        segmenter = Segmenter()

    tracker = BundleSdf(
        cfg_track_dir=cfg_track_dir,
        cfg_nerf_dir=cfg_nerf_dir,
        start_nerf_keyframes=5,
        use_gui=args.use_gui,
    )

    for i in range(0, len(reader.color_files), args.stride):
        color_file = reader.color_files[i]
        color = cv2.imread(color_file)
        h0, w0 = color.shape[:2]
        depth = reader.get_depth(i)
        h, w = depth.shape[:2]
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        if i == 0:
            mask = reader.get_mask(0)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if args.use_segmenter:
                mask = segmenter.run(color_file.replace("rgb", "masks"))
        else:
            if args.use_segmenter:
                mask = segmenter.run(color_file.replace("rgb", "masks"))
            else:
                mask = reader.get_mask(i)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if cfg_bundletrack["erode_mask"] > 0:
            kernel = np.ones(
                (cfg_bundletrack["erode_mask"], cfg_bundletrack["erode_mask"]), np.uint8
            )
            mask = cv2.erode(mask.astype(np.uint8), kernel)

        if mask.ndim == 3:
            mask = (mask.sum(axis=-1) > 0).astype(np.uint8)

        id_str = reader.id_strs[i]
        pose_in_model = np.eye(4)
        K = reader.K.copy()
        tracker.run(
            color,
            depth,
            K,
            id_str,
            mask=mask,
            occ_mask=None,
            pose_in_model=pose_in_model,
        )

    tracker.on_finish()
    print(f"Done {video_dir}")
    return "done"


def run_all():
    video_dirs = discover_video_dirs(args.video_root)
    if len(video_dirs) == 0:
        raise RuntimeError(f"No YCBInEOAT sequences found under: {args.video_root}")

    print(f"Found {len(video_dirs)} sequences")
    skipped_videos = []
    failed_videos = []
    done_videos = []
    for video_dir in video_dirs:
        try:
            status = run_one_video(video_dir, out_dir=args.out_dir)
            if status == "skipped":
                skipped_videos.append(video_dir)
            else:
                done_videos.append(video_dir)
        except Exception as e:
            print(f"Error running {video_dir}: {e}")
            failed_videos.append(video_dir)
            continue

    print("\n===== Run Summary =====")
    print(f"Done: {len(done_videos)}")
    print(f"Skipped: {len(skipped_videos)}")
    print(f"Failed: {len(failed_videos)}")
    if len(skipped_videos) > 0:
        print("Skipped sequences:")
        for video_dir in skipped_videos:
            print(f"  - {video_dir}")
    if len(failed_videos) > 0:
        print("Failed sequences:")
        for video_dir in failed_videos:
            print(f"  - {video_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dirs",
        type=str,
        default="",
        help="Comma-separated sequence dirs. Empty means running all under --video_root.",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="/home/justin/data/YCBInEOAT",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/bundlesdf/ycbineoat",
    )
    parser.add_argument(
        "--track_cfg",
        type=str,
        default=f"{SCRIPT_DIR}/BundleTrack/config_ho3d.yml",
        help="BundleTrack yaml path. Default matches run_custom.py behavior.",
    )
    parser.add_argument("--use_segmenter", type=int, default=0)
    parser.add_argument("--use_gui", type=int, default=1)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="interval of frames to run; 1 means using every frame",
    )
    parser.add_argument(
        "--shorter_side",
        type=int,
        default=480,
        help="resize input stream so min(H, W) equals this value",
    )
    parser.add_argument(
        "--debug_level", type=int, default=2, help="higher means more logging"
    )
    args = parser.parse_args()

    if args.video_dirs.strip():
        video_dirs = [x.strip() for x in args.video_dirs.split(",") if x.strip()]
        print("video_dirs:\n", video_dirs)
        skipped_videos = []
        failed_videos = []
        done_videos = []
        for video_dir in video_dirs:
            try:
                status = run_one_video(video_dir, args.out_dir)
                if status == "skipped":
                    skipped_videos.append(video_dir)
                else:
                    done_videos.append(video_dir)
            except Exception as e:
                print(f"Error running {video_dir}: {e}")
                failed_videos.append(video_dir)

        print("\n===== Run Summary =====")
        print(f"Done: {len(done_videos)}")
        print(f"Skipped: {len(skipped_videos)}")
        print(f"Failed: {len(failed_videos)}")
        if len(skipped_videos) > 0:
            print("Skipped sequences:")
            for video_dir in skipped_videos:
                print(f"  - {video_dir}")
        if len(failed_videos) > 0:
            print("Failed sequences:")
            for video_dir in failed_videos:
                print(f"  - {video_dir}")
    else:
        run_all()
