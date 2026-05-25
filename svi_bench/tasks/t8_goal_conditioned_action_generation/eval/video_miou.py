"""
Task2 Last-Frame mIoU metric.

Only evaluates on the LAST frame (frame 80) for the target players
specified in captions.json player_specifications (end_bbox).

GT comes from captions.json end_bbox (normalized [0,1]).
Pred comes from MixSort tracking results on generated videos.

Usage:
    python eval/video_miou.py \
        --video_dir /path/to/generated/videos \
        --gt_list $SVI_BENCH_DATA/T8/basketball/splits/test_100.bbox_paths.txt \
        --captions_json $SVI_BENCH_DATA/T8/basketball/captions.json \
        --eval_results_dir /path/to/eval_results
"""

import argparse
import os
import os.path as osp
import sys
import json
import csv
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miou_metric import load_bbox_file, establish_track_id_mapping

LAST_FRAME = 80


def make_parser():
    parser = argparse.ArgumentParser("Task2 Last-Frame mIoU")
    parser.add_argument("--video_dir", required=True, type=str)
    parser.add_argument("--gt_list", required=True, type=str,
                        help="test_*.bbox_paths.txt with mixsort bbox paths")
    parser.add_argument("--captions_json", required=True, type=str,
                        help="captions.json (HF-shipped, per-clip end_bbox + metadata)")
    parser.add_argument("--eval_results_dir", type=str, default=None,
                        help="MixSort tracking results dir. If None, uses GT as pred.")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_videos", type=int, default=None)
    return parser


def build_captions_lookup(json_path):
    with open(json_path) as f:
        data = json.load(f)
    entries = {}
    for mp4_key, entry in data.items():
        if "/22-23/" in mp4_key:
            rel = mp4_key.split("22-23/", 1)[1]
        elif "/clips/" in mp4_key:
            rel = mp4_key.split("clips/", 1)[1]
        else:
            continue
        rel_no_ext = osp.splitext(rel)[0]
        entries[rel_no_ext] = entry
    return entries


def mixsort_path_to_rel(mixsort_path):
    normalized = osp.normpath(mixsort_path)
    parts = normalized.split(os.sep)
    for i, part in enumerate(parts):
        if 'mixsort_all' in part:
            relative_parts = parts[i + 1:]
            rel = osp.join(*relative_parts) if relative_parts else ""
            return osp.splitext(rel)[0]
    return None


def build_gt_mapping(gt_list_path):
    """basename -> mixsort_bbox_path"""
    mapping = {}
    with open(gt_list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            basename = osp.splitext(osp.basename(line))[0]
            mapping[basename] = line
    return mapping


def get_end_bboxes(player_specs):
    """Extract end_bbox for each player as [x1, y1, x2, y2, player_idx]."""
    boxes = []
    for i, spec in enumerate(player_specs):
        end = spec.get("end_bbox", {})
        x1 = end.get("x1", 0)
        y1 = end.get("y1", 0)
        x2 = end.get("x2", 0)
        y2 = end.get("y2", 0)
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2, i])
    return boxes


def _sanitize_name(name):
    """Remove apostrophes to match MixSort output filenames."""
    return name.replace("'", "")


def find_tracking_result(eval_results_dir, basename):
    for name in (basename, _sanitize_name(basename)):
        patterns = [
            osp.join(eval_results_dir, f"gpu*/{name}/{name}.txt"),
            osp.join(eval_results_dir, f"{name}/{name}.txt"),
        ]
        for pat in patterns:
            matches = glob.glob(pat)
            if matches:
                return matches[0]
    return None


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_last_frame_miou(pred_boxes_last, gt_boxes_last, iou_threshold):
    """
    Compute mIoU at last frame only.

    Args:
        pred_boxes_last: list of [x1,y1,x2,y2,track_id] at last frame
        gt_boxes_last:   list of [x1,y1,x2,y2,player_idx] (GT end_bbox)
        iou_threshold:   min IoU for matching

    Returns:
        dict with last_frame_miou and details
    """
    if not gt_boxes_last:
        return None

    if not pred_boxes_last:
        # No predictions at last frame → 0 IoU for all GT players
        return {
            "last_frame_miou": 0.0,
            "num_gt_players": len(gt_boxes_last),
            "num_matched": 0,
            "iou_scores": [0.0] * len(gt_boxes_last),
        }

    # Match each GT player to best pred box by IoU
    iou_matrix = np.zeros((len(gt_boxes_last), len(pred_boxes_last)))
    for i, gt_box in enumerate(gt_boxes_last):
        for j, pred_box in enumerate(pred_boxes_last):
            iou_matrix[i, j] = compute_iou(gt_box, pred_box)

    # Greedy matching: for each GT player, find best unmatched pred
    matched_pred = set()
    iou_scores = []
    num_matched = 0

    for i in range(len(gt_boxes_last)):
        best_j = -1
        best_iou = 0.0
        for j in range(len(pred_boxes_last)):
            if j not in matched_pred and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j

        if best_j >= 0 and best_iou >= iou_threshold:
            iou_scores.append(best_iou)
            matched_pred.add(best_j)
            num_matched += 1
        else:
            iou_scores.append(0.0)

    last_frame_miou = float(np.mean(iou_scores)) if iou_scores else 0.0

    return {
        "last_frame_miou": last_frame_miou,
        "num_gt_players": len(gt_boxes_last),
        "num_matched": num_matched,
        "iou_scores": iou_scores,
    }


def main():
    args = make_parser().parse_args()
    output_dir = args.output_dir or osp.join(args.video_dir, "video_miou_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading polished captions from: {args.captions_json}")
    entries = build_captions_lookup(args.captions_json)
    print(f"  Loaded {len(entries)} entries")

    gt_mapping = build_gt_mapping(args.gt_list)
    print(f"GT entries: {len(gt_mapping)}")

    video_files = sorted([f for f in os.listdir(args.video_dir) if f.endswith(".mp4")])
    print(f"Found {len(video_files)} videos in {args.video_dir}")

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
        # Get GT end_bbox from polished captions
        rel_key = mixsort_path_to_rel(gt_bbox_path)
        if rel_key is None:
            failed.append(basename)
            continue

        entry = entries.get(rel_key)
        if entry is None:
            failed.append(basename)
            continue

        player_specs = entry.get("player_specifications")
        if not player_specs:
            failed.append(basename)
            continue

        gt_boxes_last = get_end_bboxes(player_specs)
        if not gt_boxes_last:
            failed.append(basename)
            continue

        # Get pred boxes at last frame
        if args.eval_results_dir:
            pred_bbox_path = find_tracking_result(args.eval_results_dir, basename)
            if not pred_bbox_path:
                failed.append(basename)
                continue
            try:
                pred_bboxes = load_bbox_file(pred_bbox_path)
            except Exception as e:
                print(f"  Error loading pred for {basename}: {e}")
                failed.append(basename)
                continue
            pred_boxes_last = pred_bboxes.get(LAST_FRAME, [])
        else:
            # GT mode: use end_bbox as both pred and GT → IoU = 1.0
            pred_boxes_last = [[b[0], b[1], b[2], b[3], b[4]] for b in gt_boxes_last]

        try:
            metrics = compute_last_frame_miou(pred_boxes_last, gt_boxes_last, args.iou_threshold)
            if metrics is None:
                failed.append(basename)
                continue

            all_metrics.append({"name": basename, **metrics})

            if (idx + 1) % 20 == 0 or idx == len(matched) - 1:
                print(f"[{idx+1}/{len(matched)}] {basename}  "
                      f"last_frame_miou={metrics['last_frame_miou']:.4f}  "
                      f"matched={metrics['num_matched']}/{metrics['num_gt_players']}")

        except Exception as e:
            print(f"[{idx+1}/{len(matched)}] {basename} ERROR: {e}")
            failed.append(basename)

    if not all_metrics:
        print("No videos evaluated!")
        return

    miou_vals = [m["last_frame_miou"] for m in all_metrics]
    summary = {
        "total_videos": len(matched),
        "evaluated": len(all_metrics),
        "failed": len(failed),
        "last_frame_miou_mean": float(np.mean(miou_vals)),
        "last_frame_miou_std": float(np.std(miou_vals)),
        "last_frame_miou_median": float(np.median(miou_vals)),
        "last_frame": LAST_FRAME,
    }

    summary_path = osp.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    csv_path = osp.join(output_dir, "per_video_metrics.csv")
    fieldnames = ["name", "last_frame_miou", "num_gt_players", "num_matched"]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in sorted(all_metrics, key=lambda x: x["name"]):
            writer.writerow(m)

    print("\n" + "=" * 60)
    print("TASK2 LAST-FRAME mIoU SUMMARY")
    print("=" * 60)
    print(f"Videos: {len(all_metrics)} / {len(matched)}")
    print(f"Last-Frame mIoU (frame {LAST_FRAME}): "
          f"{summary['last_frame_miou_mean']:.4f} "
          f"+/- {summary['last_frame_miou_std']:.4f}")
    print(f"Summary: {summary_path}")
    print(f"CSV:     {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
