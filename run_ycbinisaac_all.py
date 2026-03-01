from bundlesdf import *
import argparse
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
from BundleTrack.scripts.data_reader import *
from segmentation_utils import Segmenter


class YCBInIsaacReader:
    def __init__(self, video_dir, downscale=1, shorter_side=None):
        self.video_dir = video_dir
        self.downscale = downscale
        self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
        self.K = np.loadtxt(f"{video_dir}/cam_K.txt").reshape(3, 3)
        self.id_strs = [
            os.path.basename(color_file).replace(".png", "")
            for color_file in self.color_files
        ]
        self.H, self.W = cv2.imread(self.color_files[0]).shape[:2]

        if shorter_side is not None:
            self.downscale = shorter_side / min(self.H, self.W)

        self.H = int(self.H * self.downscale)
        self.W = int(self.W * self.downscale)
        self.K[:2] *= self.downscale

        self.masks_root = os.path.join(self.video_dir, "masks")
        self.poses_root = os.path.join(self.video_dir, "annotated_poses")
        self.object_names = self._discover_object_names()
        self.mask_files_by_object = {}
        self.gt_pose_files_by_object = {}

        if len(self.object_names) > 0:
            for obj_name in self.object_names:
                self.mask_files_by_object[obj_name] = sorted(
                    glob.glob(os.path.join(self.masks_root, obj_name, "*"))
                )
                self.gt_pose_files_by_object[obj_name] = sorted(
                    glob.glob(os.path.join(self.poses_root, obj_name, "*"))
                )
        else:
            # Backward compatibility with flat single-object layout.
            self.object_names = ["object_0"]
            self.mask_files_by_object["object_0"] = sorted(
                glob.glob(os.path.join(self.masks_root, "*"))
            )
            self.gt_pose_files_by_object["object_0"] = sorted(
                glob.glob(os.path.join(self.poses_root, "*"))
            )

    def _discover_object_names(self):
        if not os.path.isdir(self.masks_root):
            return []
        return [
            d
            for d in sorted(os.listdir(self.masks_root))
            if os.path.isdir(os.path.join(self.masks_root, d))
        ]

    @property
    def num_objects(self):
        return len(self.object_names)

    def get_video_name(self):
        return self.video_dir.split("/")[-1]

    def get_object_names(self):
        return list(self.object_names)

    def _read_and_resize_mask(self, mask_file):
        if mask_file is None or not os.path.exists(mask_file):
            return np.zeros((self.H, self.W), dtype=np.uint8)
        mask = cv2.imread(mask_file, -1)
        if mask is None:
            return np.zeros((self.H, self.W), dtype=np.uint8)
        if mask.ndim == 3:
            mask = (mask.sum(axis=-1) > 0).astype(np.uint8)
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)

    def get_mask(self, i, obj_name=None):
        if obj_name is None:
            obj_name = self.object_names[0]
        obj_mask_files = self.mask_files_by_object.get(obj_name, [])
        if len(obj_mask_files) == 0:
            return np.zeros((self.H, self.W), dtype=np.uint8)
        idx = min(i, len(obj_mask_files) - 1)
        return self._read_and_resize_mask(obj_mask_files[idx])

    def get_depth(self, i):
        depth = cv2.imread(
            self.color_files[i].replace("rgb", "depth"), cv2.IMREAD_UNCHANGED
        )
        depth = depth / 1e3
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return depth

    def get_occ_mask(self, i):
        # Keep API parity with other readers; run script currently does not pass occ_mask.
        return np.zeros((self.H, self.W), dtype=np.uint8)


def discover_video_dirs(video_root):
    video_dirs = []
    for video_dir in sorted(glob.glob(f"{video_root}/*")):
        if not os.path.isdir(video_dir):
            continue
        if len(glob.glob(f"{video_dir}/rgb/*.png")) == 0:
            continue
        video_dirs.append(video_dir)
    return video_dirs


def build_configs(video_dir, out_folder):

    cfg_bundletrack = yaml.load(open(args.track_cfg, "r"))
    cfg_bundletrack["data_dir"] = video_dir
    cfg_bundletrack["SPDLOG"] = int(args.debug_level)
    # cfg_bundletrack["depth_processing"]["percentile"] = 95
    cfg_bundletrack["erode_mask"] = 0
    cfg_bundletrack["debug_dir"] = out_folder

    cfg_bundletrack["depth_processing"].setdefault("percentile", 95)
    # cfg_bundletrack["bundle"]["max_BA_frames"] = 10
    # cfg_bundletrack["bundle"]["max_optimized_feature_loss"] = 0.03
    # cfg_bundletrack["feature_corres"]["max_dist_neighbor"] = 0.02
    # cfg_bundletrack["feature_corres"]["max_normal_neighbor"] = 30
    # cfg_bundletrack["feature_corres"]["max_dist_no_neighbor"] = 0.01
    # cfg_bundletrack["feature_corres"]["max_normal_no_neighbor"] = 20
    # cfg_bundletrack["feature_corres"]["map_points"] = True
    # cfg_bundletrack["feature_corres"]["resize"] = 400
    cfg_bundletrack["feature_corres"]["rematch_after_nerf"] = False
    # cfg_bundletrack["keyframe"]["min_rot"] = 5
    cfg_bundletrack["ransac"]["inlier_dist"] = 0.01
    cfg_bundletrack["ransac"]["inlier_normal_angle"] = 20
    cfg_bundletrack["ransac"]["max_trans_neighbor"] = 0.02
    cfg_bundletrack["ransac"]["max_rot_deg_neighbor"] = 30
    cfg_bundletrack["ransac"]["max_trans_no_neighbor"] = 0.01
    cfg_bundletrack["ransac"]["max_rot_no_neighbor"] = 10
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
    return cfg_bundletrack, cfg_track_dir, cfg_nerf_dir


def is_video_object_done(out_folder, expected_frames):
    pose_dir = f"{out_folder}/ob_in_cam"
    if not os.path.exists(pose_dir):
        return False
    pose_files = sorted(glob.glob(f"{pose_dir}/*.txt"))
    return len(pose_files) == expected_frames


def run_one_object(reader, video_dir, video_name, obj_name, out_dir, segmenter):
    out_folder = (
        f"{out_dir}/{video_name}/{obj_name}/"  # NOTE there has to be a / in the end
    )
    expected = len(range(0, len(reader.color_files), args.stride))
    if is_video_object_done(out_folder, expected):
        print(f"{out_folder} done before, skip")
        return "skipped"

    os.system(f"rm -rf {out_folder} && mkdir -p {out_folder}")
    cfg_bundletrack, cfg_track_dir, cfg_nerf_dir = build_configs(video_dir, out_folder)

    tracker = BundleSdf(
        cfg_track_dir=cfg_track_dir,
        cfg_nerf_dir=cfg_nerf_dir,
        start_nerf_keyframes=5,
        use_gui=args.use_gui,
    )

    for i in range(0, len(reader.color_files), args.stride):
        color_file = reader.color_files[i]
        color = cv2.imread(color_file)
        if color is None:
            raise RuntimeError(f"Failed to read color file: {color_file}")
        depth = reader.get_depth(i)
        h, w = depth.shape[:2]
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        if segmenter is not None:
            mask = segmenter.run(color_file.replace("rgb", "masks"))
            if mask.ndim == 3:
                mask = (mask.sum(axis=-1) > 0).astype(np.uint8)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = reader.get_mask(i, obj_name=obj_name)
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
    print(f"Done {video_dir} ({obj_name})")
    return "done"


def run_one_video(video_dir, out_dir):
    set_seed(0)
    reader = YCBInIsaacReader(video_dir=video_dir, shorter_side=args.shorter_side)
    video_name = reader.get_video_name()
    object_names = reader.get_object_names()
    print(f"[{video_name}] frames={len(reader.color_files)}, objects={object_names}")

    if len(object_names) == 0:
        print(f"No objects found under {video_dir}, skip")
        return {"done": 0, "skipped": 1, "failed": 0}

    if len(object_names) > 1:
        print(
            f"Multi-object sequence {video_name} detected "
            f"({len(object_names)} objects). Running independent BundleSDF passes."
        )
        if args.max_objects_per_sequence > 0:
            object_names = object_names[: args.max_objects_per_sequence]
            print(
                f"Using first {len(object_names)} objects due to "
                f"--max_objects_per_sequence={args.max_objects_per_sequence}: {object_names}"
            )

    segmenter = None
    if args.use_segmenter:
        if len(object_names) > 1:
            print(
                f"Warning: --use_segmenter expects a single-mask layout; "
                f"disable it for multi-object sequence {video_name}"
            )
        else:
            segmenter = Segmenter()

    stats = {"done": 0, "skipped": 0, "failed": 0}
    for obj_name in object_names:
        n_masks = len(reader.mask_files_by_object.get(obj_name, []))
        n_poses = len(reader.gt_pose_files_by_object.get(obj_name, []))
        if n_masks == 0:
            print(f"Skip {video_name}/{obj_name}: no masks found")
            stats["skipped"] += 1
            continue
        if n_poses == 0:
            print(f"Warning {video_name}/{obj_name}: no GT poses found")
        try:
            status = run_one_object(
                reader, video_dir, video_name, obj_name, out_dir, segmenter
            )
            stats[status] += 1
        except Exception as e:
            print(f"Error running {video_dir} ({obj_name}): {e}")
            stats["failed"] += 1
    return stats


def run_all():
    video_dirs = discover_video_dirs(args.video_root)
    if len(video_dirs) == 0:
        raise RuntimeError(f"No YCBInIsaac sequences found under: {args.video_root}")

    print(f"Found {len(video_dirs)} sequences")
    skipped_videos = []
    failed_videos = []
    done_videos = []
    for video_dir in video_dirs:
        stats = run_one_video(video_dir, out_dir=args.out_dir)
        if stats["failed"] > 0:
            failed_videos.append(video_dir)
        elif stats["done"] == 0 and stats["skipped"] > 0:
            skipped_videos.append(video_dir)
        else:
            done_videos.append(video_dir)

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
        default="/home/justin/data/test",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/justin/code/point-to-pose/results/bundlesdf/ycbinisaac",
    )
    parser.add_argument(
        "--track_cfg",
        type=str,
        default=f"{SCRIPT_DIR}/BundleTrack/config_ho3d.yml",
        help="BundleTrack yaml path.",
    )
    parser.add_argument("--use_segmenter", type=int, default=0)
    parser.add_argument("--use_gui", type=int, default=0)
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
    parser.add_argument(
        "--max_objects_per_sequence",
        type=int,
        default=2,
        help=(
            "If a sequence has multiple objects, run at most this many "
            "independent BundleSDF passes. Set <=0 to run all objects."
        ),
    )
    args = parser.parse_args()

    if args.video_dirs.strip():
        video_dirs = [x.strip() for x in args.video_dirs.split(",") if x.strip()]
        print("video_dirs:\n", video_dirs)
        skipped_videos = []
        failed_videos = []
        done_videos = []
        for video_dir in video_dirs:
            stats = run_one_video(video_dir, args.out_dir)
            if stats["failed"] > 0:
                failed_videos.append(video_dir)
            elif stats["done"] == 0 and stats["skipped"] > 0:
                skipped_videos.append(video_dir)
            else:
                done_videos.append(video_dir)

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
