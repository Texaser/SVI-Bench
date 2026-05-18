"""
Holistic Video mIoU metric (cf. Equation 1 in arXiv:1912.04573).

Instead of computing per-frame IoU and averaging, this accumulates
intersection and union areas across ALL frames and matched track pairs,
then computes a single IoU ratio:

    Video_mIoU = Σ_t Σ_i intersection(pred_ti, gt_ti)
                 / Σ_t Σ_i union(pred_ti, gt_ti)

This is a stricter metric that gives more weight to larger bboxes
and doesn't allow bad frames to be diluted by many good frames.

No GPU needed — pure bbox computation.

Usage:
    cd /mnt/bum/hanyi/repo/MixSort

    # Generated videos (needs tracking results from eval_generated_videos.py)
    python tools/video_miou.py \
        --video_dir /mnt/bum/hanyi/repo/ATI/samples/outputs \
        --gt_list /mnt/bum/hanyi/repo/ATI/test_subset_100.txt \
        --eval_results_dir /mnt/bum/hanyi/repo/ATI/samples/outputs/eval_results

    # GT original videos (uses GT bboxes as both pred and GT)
    python tools/video_miou.py \
        --video_dir /mnt/bum/hanyi/repo/MixSort/gt_videos_basketball \
        --gt_list /mnt/bum/hanyi/repo/sports_detection/.../test_subset.txt
"""

import argparse
import os
import os.path as osp
import sys
import json
import csv
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from miou_metric import (
    load_bbox_file,
    establish_track_id_mapping,
    find_first_appearance_frames,
)


def make_parser():
    parser = argparse.ArgumentParser("Holistic Video mIoU")
    parser.add_argument("--video_dir", required=True, type=str,
                        help="flat directory of .mp4 files (used for matching names)")
    parser.add_argument("--gt_list", required=True, type=str,
                        help="test_subset.txt with GT bbox paths")
    parser.add_argument("--eval_results_dir", type=str, default=None,
                        help="eval_generated_videos.py output dir. "
                             "If not provided, uses GT bboxes as predicted bboxes.")

    parser.add_argument("--max_frames", type=int, default=81)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--frame0_tracks_only", action="store_true", default=True)
    parser.add_argument("--no_frame0_tracks_only", dest="frame0_tracks_only",
                        action="store_false")

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_videos", type=int, default=None)

    return parser


def build_gt_mapping(gt_list_path):
    mapping = {}
    with open(gt_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            basename = osp.splitext(osp.basename(line))[0]
            mapping[basename] = line
    return mapping


def get_frame0_track_ids(gt_bbox_path):
    track_ids = set()
    with open(gt_bbox_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6 and int(parts[0]) == 0:
                track_ids.add(int(parts[1]))
    return track_ids


def load_gt_bboxes_by_frame(gt_bbox_path):
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


def find_tracking_result(eval_results_dir, basename):
    patterns = [
        osp.join(eval_results_dir, f"gpu*/{basename}/{basename}.txt"),
        osp.join(eval_results_dir, f"{basename}/{basename}.txt"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


def compute_intersection_union(box1, box2):
    """Return (intersection_area, union_area) for two [x1,y1,x2,y2] boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area, union_area


def establish_dynamic_id_mapping(pred_bboxes, gt_bboxes, iou_threshold, max_frames):
    gt_first_appearance = find_first_appearance_frames(gt_bboxes)
    all_frames = sorted(set(list(pred_bboxes.keys()) + list(gt_bboxes.keys())))
    if max_frames is not None:
        all_frames = [f for f in all_frames if f < max_frames]

    track_id_mapping = {}
    mapped_gt_ids = set()

    for frame_id in all_frames:
        gt_boxes = gt_bboxes.get(frame_id, [])
        pred_boxes = pred_bboxes.get(frame_id, [])

        new_gt_ids = set()
        for box in gt_boxes:
            gt_tid = int(box[4])
            if gt_first_appearance.get(gt_tid) == frame_id and gt_tid not in mapped_gt_ids:
                new_gt_ids.add(gt_tid)

        if new_gt_ids and pred_boxes:
            unmapped_pred = [b for b in pred_boxes if int(b[4]) not in track_id_mapping]
            new_gt_boxes = [b for b in gt_boxes if int(b[4]) in new_gt_ids]
            if unmapped_pred and new_gt_boxes:
                new_mapping = establish_track_id_mapping(unmapped_pred, new_gt_boxes, iou_threshold)
                track_id_mapping.update(new_mapping)
                mapped_gt_ids.update(new_mapping.values())

    return track_id_mapping


def compute_holistic_video_miou(pred_bboxes, gt_bboxes, track_id_mapping, max_frames):
    """Compute holistic Video mIoU by accumulating intersection and union."""
    all_frames = sorted(set(list(pred_bboxes.keys()) + list(gt_bboxes.keys())))
    if max_frames is not None:
        all_frames = [f for f in all_frames if f < max_frames]

    total_intersection = 0.0
    total_union = 0.0
    total_pairs = 0

    # Also compute per-frame average for comparison
    per_frame_ious = []

    matched_gt_ids = set(track_id_mapping.values())
    matched_pred_ids = set(track_id_mapping.keys())

    for frame_id in all_frames:
        pred_by_id = {int(b[4]): b for b in pred_bboxes.get(frame_id, [])}
        gt_by_id = {int(b[4]): b for b in gt_bboxes.get(frame_id, [])}

        frame_inter = 0.0
        frame_union = 0.0
        frame_ious = []

        for pred_id, gt_id in track_id_mapping.items():
            pred_box = pred_by_id.get(pred_id)
            gt_box = gt_by_id.get(gt_id)

            if pred_box is not None and gt_box is not None:
                # Both exist → normal IoU
                inter, union = compute_intersection_union(pred_box, gt_box)
            elif pred_box is not None and gt_box is None:
                # Case 1: pred exists but GT absent → false positive penalty
                x1, y1, x2, y2 = pred_box[:4]
                inter = 0.0
                union = (x2 - x1) * (y2 - y1)
            elif pred_box is None and gt_box is not None:
                # Case 2: GT exists but pred absent → missed detection penalty
                x1, y1, x2, y2 = gt_box[:4]
                inter = 0.0
                union = (x2 - x1) * (y2 - y1)
            else:
                # Case 3: both absent → no contribution
                continue

            total_intersection += inter
            total_union += union
            total_pairs += 1
            frame_inter += inter
            frame_union += union
            iou = inter / union if union > 0 else 0.0
            frame_ious.append(iou)

        # Unmatched GT tracks → missed detection penalty (inter=0, union=GT area)
        for gt_id, gt_box in gt_by_id.items():
            if gt_id not in matched_gt_ids:
                x1, y1, x2, y2 = gt_box[:4]
                union = (x2 - x1) * (y2 - y1)
                total_union += union
                total_pairs += 1
                frame_union += union
                frame_ious.append(0.0)

        # Unmatched pred tracks → false positive penalty (inter=0, union=pred area)
        for pred_id, pred_box in pred_by_id.items():
            if pred_id not in matched_pred_ids:
                x1, y1, x2, y2 = pred_box[:4]
                union = (x2 - x1) * (y2 - y1)
                total_union += union
                total_pairs += 1
                frame_union += union
                frame_ious.append(0.0)

        if frame_ious:
            per_frame_ious.append(np.mean(frame_ious))

    holistic_miou = total_intersection / total_union if total_union > 0 else 0.0
    avg_frame_miou = np.mean(per_frame_ious) if per_frame_ious else 0.0

    return {
        "holistic_video_miou": float(holistic_miou),
        "avg_frame_miou": float(avg_frame_miou),
        "total_intersection": float(total_intersection),
        "total_union": float(total_union),
        "total_pairs": total_pairs,
        "num_frames": len(all_frames),
        "num_tracks_matched": len(track_id_mapping),
    }


def main():
    args = make_parser().parse_args()
    video_dir = args.video_dir
    output_dir = args.output_dir or osp.join(video_dir, "video_miou_results")
    os.makedirs(output_dir, exist_ok=True)

    gt_mapping = build_gt_mapping(args.gt_list)
    print(f"GT entries: {len(gt_mapping)}")

    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    print(f"Found {len(video_files)} videos in {video_dir}")

    matched = []
    for vf in video_files:
        basename = osp.splitext(vf)[0]
        if basename in gt_mapping:
            matched.append((basename, gt_mapping[basename]))
    print(f"Matched: {len(matched)} videos")

    if args.num_videos:
        matched = matched[:args.num_videos]

    all_metrics = []
    failed = []

    for idx, (basename, gt_bbox_path) in enumerate(matched):
        # Load bboxes
        if args.eval_results_dir:
            pred_bbox_path = find_tracking_result(args.eval_results_dir, basename)
            if not pred_bbox_path:
                failed.append(basename)
                continue
        else:
            pred_bbox_path = gt_bbox_path

        try:
            pred_bboxes = load_bbox_file(pred_bbox_path)
            gt_bboxes = load_gt_bboxes_by_frame(gt_bbox_path)

            # Filter to frame-0 tracks only
            if args.frame0_tracks_only:
                frame0_ids = get_frame0_track_ids(gt_bbox_path)
                gt_bboxes = {
                    fid: [b for b in boxes if int(b[4]) in frame0_ids]
                    for fid, boxes in gt_bboxes.items()
                }
                gt_bboxes = {fid: boxes for fid, boxes in gt_bboxes.items() if boxes}

            if args.eval_results_dir is None:
                all_gt_ids = set()
                for boxes in gt_bboxes.values():
                    for b in boxes:
                        all_gt_ids.add(int(b[4]))
                track_id_mapping = {tid: tid for tid in all_gt_ids}
            else:
                track_id_mapping = establish_dynamic_id_mapping(
                    pred_bboxes, gt_bboxes, args.iou_threshold, args.max_frames
                )

            if not track_id_mapping:
                failed.append(basename)
                continue

            metrics = compute_holistic_video_miou(
                pred_bboxes, gt_bboxes, track_id_mapping, args.max_frames
            )
            all_metrics.append({"name": basename, **metrics})

            if (idx + 1) % 100 == 0 or idx == len(matched) - 1:
                print(f"[{idx+1}/{len(matched)}] {basename}  "
                      f"holistic={metrics['holistic_video_miou']:.4f}  "
                      f"avg_frame={metrics['avg_frame_miou']:.4f}")

        except Exception as e:
            print(f"[{idx+1}/{len(matched)}] {basename} ERROR: {e}")
            failed.append(basename)

    if not all_metrics:
        print("No videos evaluated!")
        return

    # Aggregate
    holistic_vals = [m["holistic_video_miou"] for m in all_metrics]
    avg_frame_vals = [m["avg_frame_miou"] for m in all_metrics]

    summary = {
        "total_videos": len(matched),
        "evaluated": len(all_metrics),
        "failed": len(failed),
        "holistic_video_miou_mean": float(np.mean(holistic_vals)),
        "holistic_video_miou_std": float(np.std(holistic_vals)),
        "holistic_video_miou_median": float(np.median(holistic_vals)),
        "avg_frame_miou_mean": float(np.mean(avg_frame_vals)),
        "avg_frame_miou_std": float(np.std(avg_frame_vals)),
        "avg_frame_miou_median": float(np.median(avg_frame_vals)),
    }

    summary_path = osp.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    csv_path = osp.join(output_dir, "per_video_metrics.csv")
    fieldnames = [
        "name", "holistic_video_miou", "avg_frame_miou",
        "total_pairs", "num_frames", "num_tracks_matched",
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in sorted(all_metrics, key=lambda x: x["name"]):
            writer.writerow(m)

    print("\n" + "=" * 60)
    print("VIDEO mIoU SUMMARY")
    print("=" * 60)
    print(f"Videos: {len(all_metrics)} / {len(matched)}")
    print(f"Holistic Video mIoU: {summary['holistic_video_miou_mean']:.4f} "
          f"+/- {summary['holistic_video_miou_std']:.4f}")
    print(f"Avg Frame mIoU:      {summary['avg_frame_miou_mean']:.4f} "
          f"+/- {summary['avg_frame_miou_std']:.4f}")
    print(f"Summary: {summary_path}")
    print(f"CSV:     {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
