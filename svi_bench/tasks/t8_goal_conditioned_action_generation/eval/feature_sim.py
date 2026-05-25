"""
T8 SigLIP2 last-frame feature similarity.

Only evaluates the LAST frame (frame 80) for target players specified by
the `player_specifications.end_bbox` field in `captions.json`.

For each target player at frame 80:
  - Find the best-matching predicted bbox (from MixSort tracking) by IoU.
  - If IoU >= threshold: crop generated video at pred bbox, crop GT video
    at GT end_bbox, compute SigLIP2 cosine similarity.
  - If no match: similarity = 0.

Also reports detection_rate = #matched_players / #total_target_players, and
last_frame_miou for convenience.

All DINOv3 code paths from the upstream variant were removed; the paper
reports SigLIP, and SigLIP2 is its successor.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import os.path as osp
import re
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

LAST_FRAME = 80


def make_parser():
    p = argparse.ArgumentParser("T8 SigLIP2 last-frame feature similarity")
    p.add_argument("--video_dir", default=None,
                   help="Dir containing generated .mp4 files (flat or subdir/generated.mp4)")
    p.add_argument("--gt_list", default=None,
                   help="test_*.bbox_paths.txt with mixsort bbox paths")
    p.add_argument("--captions_json", default=None,
                   help="captions.json (HF-shipped, per-clip end_bbox + caption metadata)")
    p.add_argument("--eval_results_dir", default=None,
                   help="MixSort tracking results dir (gpu*/{clip}/...)")
    p.add_argument("--sport", default="basketball", choices=["basketball", "soccer"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--aggregate_only", action="store_true")

    p.add_argument("--siglip_model", default="google/siglip2-so400m-patch14-384")
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--min_crop_size", type=int, default=10)

    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--num_clips", type=int, default=None)
    return p


# ---------------------------------------------------------------------------
# Path / data helpers
# ---------------------------------------------------------------------------

def bbox_path_to_video_path(bbox_path, sport):
    # bbox: .../bboxes/{bucket}/{ID}.txt -> mp4: .../clips/{bucket}/{ID}.mp4
    return re.sub(r"\.txt$", ".mp4", bbox_path.replace("/bboxes/", "/clips/", 1))


def build_gt_mapping(gt_list_path):
    m = {}
    with open(gt_list_path) as f:
        for line in f:
            line = line.strip()
            if line:
                m[osp.splitext(osp.basename(line))[0]] = line
    return m


def build_captions_lookup(json_path):
    """Returns dict: sample_id -> entry (captions.json is keyed by anon ID)."""
    with open(json_path) as f:
        return json.load(f)


def bbox_path_to_id(bbox_path):
    """Return the anon sample ID from a bbox path (basename without extension)."""
    return osp.splitext(osp.basename(bbox_path))[0]


def _sanitize_name(name):
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


def get_end_bboxes(player_specs):
    boxes = []
    for i, spec in enumerate(player_specs):
        end = spec.get("end_bbox", {})
        x1, y1 = end.get("x1", 0), end.get("y1", 0)
        x2, y2 = end.get("x2", 0), end.get("y2", 0)
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2, i])
    return boxes


def load_bbox_file(path):
    bboxes = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            tid = int(parts[1])
            x1, y1, x2, y2 = map(float, parts[2:6])
            bboxes.setdefault(fid, []).append([x1, y1, x2, y2, tid])
    return bboxes


def compute_iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def crop_player(frame_bgr, bbox_norm, img_w, img_h, min_size=10):
    x1 = max(0, int(bbox_norm[0] * img_w))
    y1 = max(0, int(bbox_norm[1] * img_h))
    x2 = min(img_w, int(bbox_norm[2] * img_w))
    y2 = min(img_h, int(bbox_norm[3] * img_h))
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None
    return cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# SigLIP2 model
# ---------------------------------------------------------------------------

def load_siglip(model_name, device):
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(model_name).vision_model.to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def extract_siglip(crops, model, processor, device, batch_size):
    if not crops:
        return torch.zeros(0)
    use_amp = device.type == "cuda"
    feats = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(pixel_values=pv)
            f = F.normalize(out.pooler_output.float(), p=2, dim=-1)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(output_dir):
    json_files = sorted(glob.glob(osp.join(output_dir, "*.json")))
    json_files = [f for f in json_files if osp.basename(f) != "summary.json"]

    all_metrics = []
    for jf in json_files:
        try:
            with open(jf) as f:
                m = json.load(f)
            m["clip"] = osp.splitext(osp.basename(jf))[0]
            all_metrics.append(m)
        except Exception as e:
            logger.warning(f"Could not read {jf}: {e}")

    if not all_metrics:
        logger.error("No per-clip JSONs found for aggregation.")
        return

    def mean_std_med(key):
        vals = [m[key] for m in all_metrics if key in m and m[key] is not None]
        if not vals:
            return 0.0, 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals)), float(np.median(vals))

    summary = {"evaluated": len(all_metrics)}
    for metric, label in [
        ("siglip_sim", "siglip"),
        ("detection_rate", "detection_rate"),
        ("last_frame_miou", "last_frame_miou"),
    ]:
        mu, sd, med = mean_std_med(metric)
        summary[f"{label}_mean"] = mu
        summary[f"{label}_std"] = sd
        summary[f"{label}_median"] = med

    summary_path = osp.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = osp.join(output_dir, "per_clip_metrics.csv")
    fieldnames = ["clip", "siglip_sim", "last_frame_miou",
                  "detection_rate", "num_gt_players", "num_matched"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for m in sorted(all_metrics, key=lambda x: x["clip"]):
            w.writerow(m)

    logger.info("\n" + "=" * 60)
    logger.info("T8 LAST-FRAME FEATURE SIMILARITY (SigLIP2) SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Clips evaluated: {len(all_metrics)}")
    logger.info(f"  SigLIP sim : {summary['siglip_mean']:.4f} +/- {summary['siglip_std']:.4f}  "
                f"(median={summary['siglip_median']:.4f})")
    logger.info(f"  Det. rate  : {summary['detection_rate_mean']:.4f} +/- {summary['detection_rate_std']:.4f}")
    logger.info(f"  mIoU       : {summary['last_frame_miou_mean']:.4f} +/- {summary['last_frame_miou_std']:.4f}")
    logger.info(f"Summary: {summary_path}")
    logger.info(f"CSV:     {csv_path}")
    logger.info("=" * 60)


def main():
    args = make_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.aggregate_only:
        aggregate_results(args.output_dir)
        return

    assert args.video_dir and args.gt_list and args.captions_json and args.eval_results_dir, \
        "--video_dir, --gt_list, --captions_json, --eval_results_dir required"

    logger.info(f"Loading polished captions from: {args.captions_json}")
    captions = build_captions_lookup(args.captions_json)
    logger.info(f"  Loaded {len(captions)} entries")

    gt_mapping = build_gt_mapping(args.gt_list)
    logger.info(f"GT entries: {len(gt_mapping)}")

    video_files = sorted([f for f in os.listdir(args.video_dir) if f.endswith(".mp4")])
    logger.info(f"Found {len(video_files)} videos in {args.video_dir}")

    matched = []
    for vf in video_files:
        basename = osp.splitext(vf)[0]
        if basename in gt_mapping:
            matched.append((basename, gt_mapping[basename]))

    logger.info(f"Matched: {len(matched)} videos")
    if args.num_clips:
        matched = matched[:args.num_clips]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading SigLIP2...")
    model_siglip, proc_siglip = load_siglip(args.siglip_model, device)
    logger.info("SigLIP2 loaded.")

    failed = []
    t_start = time.time()

    for idx, (basename, gt_bbox_path) in enumerate(matched):
        out_json = osp.join(args.output_dir, f"{basename}.json")
        if args.skip_existing and osp.exists(out_json):
            continue

        entry = captions.get(bbox_path_to_id(gt_bbox_path))
        if entry is None:
            failed.append(basename); continue

        player_specs = entry.get("player_specifications")
        if not player_specs:
            failed.append(basename); continue

        gt_end_bboxes = get_end_bboxes(player_specs)
        if not gt_end_bboxes:
            failed.append(basename); continue

        pred_path = find_tracking_result(args.eval_results_dir, basename)
        if not pred_path:
            failed.append(basename); continue

        try:
            pred_bboxes = load_bbox_file(pred_path)
        except Exception as e:
            logger.error(f"Error loading pred for {basename}: {e}")
            failed.append(basename); continue

        pred_boxes_last = pred_bboxes.get(LAST_FRAME, [])

        gt_video_path = bbox_path_to_video_path(gt_bbox_path, args.sport)
        if gt_video_path is None or not osp.exists(gt_video_path):
            logger.warning(f"[{idx+1}] GT video not found: {gt_video_path}")
            failed.append(basename); continue

        gen_video_path = osp.join(args.video_dir, f"{basename}.mp4")
        if not osp.isfile(gen_video_path):
            gen_video_path = osp.join(args.video_dir, basename, "generated.mp4")
        if not osp.isfile(gen_video_path):
            logger.warning(f"[{idx+1}] Generated video not found for {basename}")
            failed.append(basename); continue

        try:
            cap_gen = cv2.VideoCapture(gen_video_path)
            cap_gt = cv2.VideoCapture(gt_video_path)
            if not cap_gen.isOpened() or not cap_gt.isOpened():
                logger.error(f"Cannot open video(s) for {basename}")
                failed.append(basename); continue

            W_gen = int(cap_gen.get(cv2.CAP_PROP_FRAME_WIDTH))
            H_gen = int(cap_gen.get(cv2.CAP_PROP_FRAME_HEIGHT))
            W_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_WIDTH))
            H_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cap_gen.set(cv2.CAP_PROP_POS_FRAMES, LAST_FRAME)
            cap_gt.set(cv2.CAP_PROP_POS_FRAMES, LAST_FRAME)
            ret_gen, frm_gen = cap_gen.read()
            ret_gt, frm_gt = cap_gt.read()
            cap_gen.release()
            cap_gt.release()
            if not ret_gen or not ret_gt:
                logger.error(f"Cannot read frame {LAST_FRAME} for {basename}")
                failed.append(basename); continue

            gen_crops, gt_crops, matched_flags, iou_scores = [], [], [], []
            matched_pred_set = set()

            for gt_box in gt_end_bboxes:
                best_iou, best_pred_idx = 0.0, -1
                for j, pb in enumerate(pred_boxes_last):
                    if j in matched_pred_set:
                        continue
                    iou = compute_iou(gt_box[:4], pb[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = j

                if best_iou >= args.iou_threshold and best_pred_idx >= 0:
                    matched_pred_set.add(best_pred_idx)
                    iou_scores.append(best_iou)
                    pred_box = pred_boxes_last[best_pred_idx]
                    gen_crop = crop_player(frm_gen, pred_box[:4], W_gen, H_gen, args.min_crop_size)
                    gt_crop = crop_player(frm_gt, gt_box[:4], W_gt, H_gt, args.min_crop_size)
                    if gen_crop is not None and gt_crop is not None:
                        gen_crops.append(gen_crop)
                        gt_crops.append(gt_crop)
                        matched_flags.append(True)
                    else:
                        matched_flags.append(False)
                        iou_scores[-1] = 0.0
                else:
                    matched_flags.append(False)
                    iou_scores.append(0.0)

            num_gt = len(gt_end_bboxes)
            num_matched = sum(matched_flags)
            detection_rate = num_matched / num_gt if num_gt > 0 else 0.0
            last_frame_miou = float(np.mean(iou_scores)) if iou_scores else 0.0

            if gen_crops:
                all_crops = gen_crops + gt_crops
                n = len(gen_crops)
                sig_feats = extract_siglip(all_crops, model_siglip, proc_siglip,
                                           device, args.batch_size)
                sig_sims = F.cosine_similarity(sig_feats[:n], sig_feats[n:], dim=-1).numpy()
                full_sig = np.zeros(num_gt)
                crop_idx = 0
                for i, m in enumerate(matched_flags):
                    if m:
                        full_sig[i] = sig_sims[crop_idx]
                        crop_idx += 1
                siglip_sim = float(full_sig.mean())
            else:
                siglip_sim = 0.0

            metrics = {
                "siglip_sim": siglip_sim,
                "last_frame_miou": last_frame_miou,
                "detection_rate": detection_rate,
                "num_gt_players": num_gt,
                "num_matched": num_matched,
            }
            with open(out_json, "w") as f:
                json.dump(metrics, f, indent=2)

            if (idx + 1) % 20 == 0 or idx == len(matched) - 1:
                logger.info(
                    f"[{idx+1}/{len(matched)}] {basename}  "
                    f"sig={siglip_sim:.4f}  miou={last_frame_miou:.4f}  "
                    f"det={detection_rate:.2f}  ({num_matched}/{num_gt})"
                )
        except Exception as e:
            logger.error(f"[{idx+1}] ERROR {basename}: {e}")
            import traceback; traceback.print_exc()
            failed.append(basename)

    elapsed = time.time() - t_start
    logger.info(f"Done in {elapsed:.1f}s. Failed: {len(failed)}")
    aggregate_results(args.output_dir)


if __name__ == "__main__":
    main()
