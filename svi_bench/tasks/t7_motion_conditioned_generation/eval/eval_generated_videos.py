"""
Batch mIoU evaluation worker.

Evaluates a flat directory of generated .mp4 files against GT bounding boxes.
Runs MixSort tracking on each video and computes mIoU. Invoked by
`eval/run_basketball.sh` and `eval/run_soccer.sh` (one process per GPU).

Usage:
    python eval/eval_generated_videos.py \
        --video_dir /path/to/your/generated/videos \
        --gt_list $SVI_BENCH_DATA/T7/basketball/splits/test_subset_100.bbox_paths.txt \
        --exp_file eval/exps/example/mot/yolox_x_sportsmot.py \
        --ckpt eval/pretrained/yolox_x_sports_train.pth.tar \
        --output_dir <video_dir>/eval_results
"""

import argparse
import os
import os.path as osp
import sys
import json
import time
import csv
import cv2
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'MixViT'))

from loguru import logger

from yolox.exp import get_exp
from yolox.utils import fuse_model

from miou_metric import (
    Predictor,
    imageflow_demo_allframes,
    compute_video_miou,
)


def make_parser():
    parser = argparse.ArgumentParser("Evaluate generated videos (ATI / MagicMotion)")

    # Required paths
    parser.add_argument("--video_dir", required=True, type=str,
                        help="directory containing generated .mp4 files (flat)")
    parser.add_argument("--gt_list", required=True, type=str,
                        help="path to test_subset.txt with GT bbox paths")

    # Model config
    parser.add_argument("-f", "--exp_file", required=True, type=str,
                        help="experiment description file")
    parser.add_argument("-c", "--ckpt", required=True, type=str,
                        help="checkpoint path")
    parser.add_argument("--device", default="gpu", type=str)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fuse", action="store_true")

    # Detection params
    parser.add_argument("--conf", default=0.1, type=float)
    parser.add_argument("--nms", default=0.7, type=float)
    parser.add_argument("--tsize", default=None, type=int)

    # Tracking params
    parser.add_argument("--track_thresh", type=float, default=0.6)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.9)
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
    parser.add_argument("--min_box_area", type=float, default=10)

    # MixSort specific
    parser.add_argument("--script", type=str, default='mixformer_deit')
    parser.add_argument("--config", type=str, default='track')
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--radius", type=int, default=0)
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument("--iou_only", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)

    # mIoU params
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for ID mapping")
    parser.add_argument("--penalize_unmatched", action="store_true", default=True)
    parser.add_argument("--no_penalize_unmatched", dest="penalize_unmatched",
                        action="store_false")
    parser.add_argument("--max_frames", type=int, default=81,
                        help="only evaluate the first N frames")

    # Execution control
    parser.add_argument("--output_dir", type=str, default=None,
                        help="output dir for results (default: video_dir/eval_results)")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--verbosity", type=int, default=100,
                        help="log every N frames during tracking")
    parser.add_argument("--skip_existing", action="store_true",
                        help="skip videos that already have metrics.json")
    parser.add_argument("--num_visualize", type=int, default=5,
                        help="save annotated videos for first N videos")
    parser.add_argument("--frame0_tracks_only", action="store_true", default=True,
                        help="only evaluate GT tracks present in frame 0 (default: True for ATI/MagicMotion)")
    parser.add_argument("--no_frame0_tracks_only", dest="frame0_tracks_only",
                        action="store_false",
                        help="evaluate all GT tracks including those appearing after frame 0")

    return parser


def build_gt_mapping(gt_list_path):
    """Read test_subset.txt and map basename -> GT bbox path."""
    mapping = {}
    with open(gt_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            basename = osp.splitext(osp.basename(line))[0]
            mapping[basename] = line
    return mapping


def load_gt_bboxes_by_frame(gt_bbox_path):
    """Load GT bboxes grouped by frame.
    Returns dict {frame_id: [[x1,y1,x2,y2,track_id], ...]}."""
    bboxes_by_frame = {}
    with open(gt_bbox_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x1, y1, x2, y2 = map(float, parts[2:6])
            if frame_id not in bboxes_by_frame:
                bboxes_by_frame[frame_id] = []
            bboxes_by_frame[frame_id].append([x1, y1, x2, y2, track_id])
    return bboxes_by_frame


def get_frame0_track_ids(gt_bbox_path):
    """Get the set of track_ids that appear in frame 0."""
    track_ids = set()
    with open(gt_bbox_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            if frame_id == 0:
                track_ids.add(int(parts[1]))
    return track_ids


def write_filtered_gt(gt_bbox_path, output_path, allowed_track_ids):
    """Write a filtered GT bbox file containing only allowed track_ids."""
    with open(gt_bbox_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(',')
            if len(parts) < 6:
                continue
            track_id = int(parts[1])
            if track_id in allowed_track_ids:
                fout.write(line)


def run_tracking_single(predictor, exp, args, video_path, output_dir,
                        gt_bboxes_by_frame=None, save_video=False):
    """Run MixSort tracking on a single video, return path to bbox txt."""
    source_basename = osp.splitext(osp.basename(video_path))[0]
    bbox_output_path = osp.join(output_dir, f"{source_basename}.txt")

    args.path = video_path
    args.original_path = video_path
    args.processed_path = video_path
    args.save_result = True
    args.save_video = save_video
    args.save_first_iou = False
    args.save_per_frame_iou = False
    args.video_output_dir = output_dir
    args.bbox_output_dir = output_dir

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    cap.release()

    args.processed_fps = fps
    args.processed_width = w
    args.processed_height = h
    args.clip_start_sec = None
    args.clip_end_sec = None
    args.clip_fps = None

    imageflow_demo_allframes(predictor, output_dir, args, exp,
                             gt_bboxes_by_frame=gt_bboxes_by_frame)
    return bbox_output_path


def main():
    args = make_parser().parse_args()

    video_dir = args.video_dir
    gt_list_path = args.gt_list
    output_dir = args.output_dir or osp.join(video_dir, "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    # Build GT mapping: basename -> GT bbox path
    logger.info(f"Reading GT list from: {gt_list_path}")
    gt_mapping = build_gt_mapping(gt_list_path)
    logger.info(f"Found {len(gt_mapping)} GT entries")

    # Discover video files in flat directory
    video_files = sorted([
        f for f in os.listdir(video_dir) if f.endswith(".mp4")
    ])
    logger.info(f"Found {len(video_files)} video files in {video_dir}")

    # Match videos to GT by basename
    matched = []
    unmatched = []
    for vf in video_files:
        basename = osp.splitext(vf)[0]
        if basename in gt_mapping:
            matched.append((basename, vf, gt_mapping[basename]))
        else:
            unmatched.append(vf)

    logger.info(f"Matched: {len(matched)}, Unmatched: {len(unmatched)}")
    if unmatched:
        logger.warning(f"First 5 unmatched: {unmatched[:5]}")

    # Load model ONCE
    logger.info("Loading model...")
    args.device = torch.device(
        "cuda" if (args.device == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    exp = get_exp(args.exp_file, None)
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    model.eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    logger.info(f"Model loaded from {args.ckpt}")

    if args.fuse:
        model = fuse_model(model)
    if args.fp16:
        model = model.half()

    predictor = Predictor(model, exp, args.device, args.fp16)

    # Iterate matched videos
    all_metrics = []
    failed = []
    t_start = time.time()

    for idx, (basename, video_file, gt_bbox_path) in enumerate(matched):
        video_path = osp.join(video_dir, video_file)
        vid_output_dir = osp.join(output_dir, basename)
        os.makedirs(vid_output_dir, exist_ok=True)

        logger.info(f"\n[{idx+1}/{len(matched)}] Processing: {basename}")

        # Skip if already computed
        metrics_json = osp.join(vid_output_dir, "metrics.json")
        if args.skip_existing and osp.exists(metrics_json):
            try:
                with open(metrics_json, 'r') as f:
                    m = json.load(f)
                all_metrics.append({"name": basename, **m})
                logger.info(f"  Skipped (existing), mIoU={m['video_miou']:.4f}")
                continue
            except Exception:
                pass

        if not osp.exists(gt_bbox_path):
            logger.error(f"  GT bbox not found: {gt_bbox_path}")
            failed.append(basename)
            continue

        try:
            gt_bboxes = load_gt_bboxes_by_frame(gt_bbox_path)

            # Filter GT to frame-0 tracks only (ATI/MagicMotion only condition on these)
            if args.frame0_tracks_only:
                frame0_ids = get_frame0_track_ids(gt_bbox_path)
                # Filter gt_bboxes_by_frame for tracking injection
                gt_bboxes_filtered = {}
                for fid, boxes in gt_bboxes.items():
                    filtered = [b for b in boxes if int(b[4]) in frame0_ids]
                    if filtered:
                        gt_bboxes_filtered[fid] = filtered
                gt_bboxes = gt_bboxes_filtered

                # Write filtered GT file for mIoU computation
                filtered_gt_path = osp.join(vid_output_dir, f"{basename}_gt_frame0only.txt")
                write_filtered_gt(gt_bbox_path, filtered_gt_path, frame0_ids)
                gt_path_for_miou = filtered_gt_path
            else:
                gt_path_for_miou = gt_bbox_path

            do_visualize = (args.num_visualize > 0 and idx < args.num_visualize)

            # Run tracking
            result_path = run_tracking_single(
                predictor, exp, args, video_path, vid_output_dir,
                gt_bboxes_by_frame=gt_bboxes,
                save_video=do_visualize,
            )
            if result_path is None or not osp.exists(result_path):
                logger.error(f"  Tracking failed")
                failed.append(basename)
                continue

            # Compute mIoU
            metrics = compute_video_miou(
                result_path,
                gt_path_for_miou,
                iou_threshold=args.iou_threshold,
                penalize_unmatched=args.penalize_unmatched,
                max_frames=args.max_frames,
            )

            with open(metrics_json, 'w') as f:
                json.dump(metrics, f, indent=2)

            all_metrics.append({"name": basename, **metrics})
            logger.info(
                f"  mIoU={metrics['video_miou']:.4f}  "
                f"precision={metrics['precision']:.4f}  "
                f"recall={metrics['recall']:.4f}"
            )

        except Exception as e:
            logger.error(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed.append(basename)
            continue

    elapsed = time.time() - t_start

    # Aggregate results
    if all_metrics:
        mious = [m['video_miou'] for m in all_metrics]
        precisions = [m['precision'] for m in all_metrics]
        recalls = [m['recall'] for m in all_metrics]
        f1s = [m['f1_score'] for m in all_metrics]

        summary = {
            "total_videos": len(matched),
            "evaluated": len(all_metrics),
            "failed": len(failed),
            "mean_miou": float(np.mean(mious)),
            "std_miou": float(np.std(mious)),
            "median_miou": float(np.median(mious)),
            "mean_precision": float(np.mean(precisions)),
            "mean_recall": float(np.mean(recalls)),
            "mean_f1": float(np.mean(f1s)),
            "elapsed_seconds": elapsed,
            "iou_threshold": args.iou_threshold,
            "penalize_unmatched": args.penalize_unmatched,
            "video_dir": video_dir,
        }

        summary_path = osp.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        csv_path = osp.join(output_dir, "per_video_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "name", "video_miou", "precision", "recall", "f1_score",
                "total_frames", "total_matched", "total_pred", "total_gt",
                "num_id_mappings", "num_gt_ids", "num_mapped_gt_ids",
            ])
            writer.writeheader()
            for m in all_metrics:
                writer.writerow({k: m.get(k, "") for k in writer.fieldnames})

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Video dir:        {video_dir}")
        logger.info(f"Videos evaluated: {len(all_metrics)} / {len(matched)}")
        logger.info(f"Failed:           {len(failed)}")
        logger.info(f"Mean mIoU:        {summary['mean_miou']:.4f} +/- {summary['std_miou']:.4f}")
        logger.info(f"Median mIoU:      {summary['median_miou']:.4f}")
        logger.info(f"Mean Precision:   {summary['mean_precision']:.4f}")
        logger.info(f"Mean Recall:      {summary['mean_recall']:.4f}")
        logger.info(f"Mean F1:          {summary['mean_f1']:.4f}")
        logger.info(f"Time:             {elapsed:.1f}s ({elapsed/max(1,len(all_metrics)):.1f}s/video)")
        logger.info(f"Summary:          {summary_path}")
        logger.info(f"Per-video CSV:    {csv_path}")
        logger.info("=" * 60)

        if failed:
            failed_path = osp.join(output_dir, "failed.txt")
            with open(failed_path, 'w') as f:
                f.write("\n".join(failed) + "\n")
            logger.info(f"Failed list:      {failed_path}")
    else:
        logger.error("No videos were successfully evaluated!")


if __name__ == "__main__":
    main()
