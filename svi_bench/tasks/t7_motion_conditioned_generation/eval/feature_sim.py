"""
T7 SigLIP2 feature similarity — slim of upstream MixSort `feature_sim_v2.py`.

Compares appearance features between generated and GT videos via SigLIP2.
All DINOv3 code paths from the upstream variant were removed; the paper
reports SigLIP, and SigLIP2 is its successor.

Modes:

  iou_gated (default — Mode A in the upstream paper / docstring)
    For each GT bbox at each frame, find the predicted bbox with the
    highest IoU.
      - If IoU >= threshold: sim = cosine_sim(
            gen_crop_at_pred_box, gt_crop_at_gt_box)
      - If no match (track lost): sim = 0
    Also reports detection_rate = #frames_with_valid_match / #total_GT_box_frames.

  gt_box (Mode B)
    Skip the tracker entirely. For each GT bbox at each frame:
      - Crop the generated video at gt_bbox_transformed coords
      - Crop the GT original video at original GT bbox coords
      - sim = cosine_sim(gen_crop, gt_crop)
    Fixes survivorship bias.

  both
    Run both and report both columns.

Inputs (mIoU pipeline product):
  results_dir/<clip>/{generated.txt, gt_bbox_transformed.txt}
  video_dir/<clip>/generated.mp4   (or <clip>.mp4 flat layout)
  gt_list                          (test_subset.txt mapping clip basenames to GT bbox paths)

Usage (multi-GPU shell wrappers are in run_basketball_featsim.sh and
run_soccer_featsim.sh — invoke those rather than this script directly).
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

# Legacy DATA_ROOT, used only for the pre-anonymization input layout where
# bbox files lived in a different parallel tree from the video files. The
# anonymized layout shipped on HF (clips/<bucket>/<id>.mp4 next to
# bboxes/<bucket>/<id>.txt) is the common case and is detected first inside
# `bbox_path_to_video_path`.
DATA_ROOT = os.environ.get("SVI_LEGACY_DATA_ROOT", "")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def make_parser():
    p = argparse.ArgumentParser("T7 SigLIP2 feature similarity")
    p.add_argument("--results_dir", default=None,
                   help="Flat clip dir (miou_results_all/) or step dir with miou_results_* subdirs. "
                        "Not required when --aggregate_only is set.")
    p.add_argument("--video_dir", default=None,
                   help="Step dir where clip_name/generated.mp4 files live")
    p.add_argument("--gt_list", default=None,
                   help="test_subset.txt mapping clip basenames to GT bbox paths")
    p.add_argument("--sport", default=None, choices=["basketball", "soccer"])
    p.add_argument("--mode", default="iou_gated", choices=["iou_gated", "gt_box", "both"],
                   help="iou_gated=Mode A (default), gt_box=Mode B, both=run both")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--aggregate_only", action="store_true",
                   help="Skip inference; just read existing per-clip JSONs and recompute summary")

    p.add_argument("--siglip_model", default="google/siglip2-so400m-patch14-384")

    p.add_argument("--max_frames", type=int, default=81)
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--min_crop_size", type=int, default=10)

    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--num_clips", type=int, default=None)
    p.add_argument("--gt_baseline", action="store_true",
                   help="GT-video upper-bound baseline: use GT video as the 'generated' "
                        "video and GT tracking as pred bboxes.")
    p.add_argument("--frame0_only", action="store_true",
                   help="Only evaluate GT track IDs that appear at frame 0.")
    return p


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def bbox_path_to_video_path(bbox_path, sport):
    # Anonymized SVI-Bench layout (the common case after running
    # scripts/download_t7_t8.sh): bbox at .../bboxes/<bucket>/<id>.txt
    # has its mp4 sibling at .../clips/<bucket>/<id>.mp4.
    if "/bboxes/" in bbox_path:
        return re.sub(r"\.txt$", ".mp4", bbox_path.replace("/bboxes/", "/clips/", 1))

    # Pre-anonymization legacy layout: bbox files lived under
    # $DATA_ROOT/<sport>_mixsort_all*/... and videos under
    # $DATA_ROOT/<sport>_fps_15/...
    if not DATA_ROOT:
        return None
    rel = bbox_path.replace(DATA_ROOT, "")
    if sport == "soccer":
        rel = rel.replace("soccer_mixsort_all_filtered_10/", "soccer_video_fps_15/", 1)
    elif sport == "basketball":
        for old in ("basketball_mixsort_all_22_23_season_filter_f_8/",
                    "basketball_mixsort_all_23_24_season_filter_f_8/",
                    "basketball_mixsort_all_22_23_season/",
                    "basketball_mixsort_all_23_24_season/"):
            if rel.startswith(old):
                rel = rel.replace(old, "basketball_fps_15/", 1)
                break
        else:
            return None
    return osp.join(DATA_ROOT, re.sub(r"\.txt$", ".mp4", rel))


def build_gt_mapping(gt_list_path):
    m = {}
    with open(gt_list_path) as f:
        for line in f:
            line = line.strip()
            if line:
                m[osp.splitext(osp.basename(line))[0]] = line
    return m


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


def crop_player(frame_bgr, bbox, img_w, img_h, min_size=10):
    x1 = max(0, int(bbox[0] * img_w))
    y1 = max(0, int(bbox[1] * img_h))
    x2 = min(img_w, int(bbox[2] * img_w))
    y2 = min(img_h, int(bbox[3] * img_h))
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


def compute_sims(crops_a, crops_b, model_siglip, proc_siglip, device, batch_size):
    """Return per-pair SigLIP cosine similarities for paired crops_a / crops_b."""
    assert len(crops_a) == len(crops_b)
    all_crops = crops_a + crops_b
    n = len(crops_a)
    sig = extract_siglip(all_crops, model_siglip, proc_siglip, device, batch_size)
    return F.cosine_similarity(sig[:n], sig[n:], dim=-1).numpy()


# ---------------------------------------------------------------------------
# Per-clip computation
# ---------------------------------------------------------------------------

def process_clip(
    clip_name, gen_video_path, gt_video_path,
    gt_bbox_transformed, pred_bboxes, orig_gt_bboxes,
    model_siglip, proc_siglip, device, args,
):
    cap_gen = cv2.VideoCapture(gen_video_path)
    cap_gt = cv2.VideoCapture(gt_video_path)
    if not cap_gen.isOpened() or not cap_gt.isOpened():
        logger.error(f"Cannot open video(s) for {clip_name}")
        return None

    W_gen = int(cap_gen.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_gen = int(cap_gen.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_HEIGHT))

    a_gen_crops, a_gt_crops, a_matched = [], [], []
    a_total_gt_boxes, a_matched_boxes = 0, 0
    b_gen_crops, b_gt_crops = [], []

    frame0_ids = None
    if args.frame0_only:
        frame0_ids = {int(b[4]) for b in gt_bbox_transformed.get(0, [])}

    frame_id = 0
    while frame_id < args.max_frames:
        ret_gen, frm_gen = cap_gen.read()
        ret_gt, frm_gt = cap_gt.read()
        if not ret_gen or not ret_gt:
            break

        gt_trans_boxes = gt_bbox_transformed.get(frame_id, [])
        if frame0_ids is not None:
            gt_trans_boxes = [b for b in gt_trans_boxes if int(b[4]) in frame0_ids]
        pred_boxes_list = pred_bboxes.get(frame_id, [])
        orig_gt_list = orig_gt_bboxes.get(frame_id, [])
        orig_gt_by_id = {int(b[4]): b for b in orig_gt_list}

        for gt_box in gt_trans_boxes:
            gt_tid = int(gt_box[4])
            orig_box = orig_gt_by_id.get(gt_tid)

            # Mode B (GT-box, no tracker)
            if args.mode in ("gt_box", "both"):
                gen_crop = crop_player(frm_gen, gt_box[:4], W_gen, H_gen, args.min_crop_size)
                gt_crop = crop_player(frm_gt, orig_box[:4], W_gt, H_gt, args.min_crop_size) \
                    if orig_box is not None else None
                if gen_crop is not None and gt_crop is not None:
                    b_gen_crops.append(gen_crop)
                    b_gt_crops.append(gt_crop)

            # Mode A (IoU-gated, tracker)
            if args.mode in ("iou_gated", "both"):
                a_total_gt_boxes += 1
                best_iou, best_pred = 0.0, None
                for pb in pred_boxes_list:
                    iou = compute_iou(gt_box[:4], pb[:4])
                    if iou > best_iou:
                        best_iou, best_pred = iou, pb

                if best_iou >= args.iou_threshold and best_pred is not None:
                    a_matched_boxes += 1
                    gen_crop_a = crop_player(frm_gen, best_pred[:4], W_gen, H_gen, args.min_crop_size)
                    gt_crop_a = crop_player(frm_gt, orig_box[:4], W_gt, H_gt, args.min_crop_size) \
                        if orig_box is not None else None
                    if gen_crop_a is not None and gt_crop_a is not None:
                        a_gen_crops.append(gen_crop_a)
                        a_gt_crops.append(gt_crop_a)
                        a_matched.append(True)
                    else:
                        a_matched.append(False)
                else:
                    a_matched.append(False)

        frame_id += 1

    cap_gen.release()
    cap_gt.release()

    result = {}

    if args.mode in ("iou_gated", "both"):
        result["detection_rate"] = float(a_matched_boxes / a_total_gt_boxes) if a_total_gt_boxes else 0.0
        result["total_gt_boxes"] = a_total_gt_boxes
        result["matched_boxes"] = a_matched_boxes
        if a_gen_crops:
            sig_sims = compute_sims(a_gen_crops, a_gt_crops,
                                    model_siglip, proc_siglip, device, args.batch_size)
            full_sig = np.zeros(len(a_matched))
            matched_idx = [i for i, m in enumerate(a_matched) if m]
            valid_idx = matched_idx[:len(sig_sims)]
            full_sig[valid_idx] = sig_sims[:len(valid_idx)]
            result["iou_gated_siglip_mean"] = float(full_sig.mean())
            result["iou_gated_siglip_std"] = float(full_sig.std())
            result["iou_gated_num_pairs"] = len(a_gen_crops)
        else:
            result["iou_gated_siglip_mean"] = 0.0
            result["iou_gated_siglip_std"] = 0.0
            result["iou_gated_num_pairs"] = 0

    if args.mode in ("gt_box", "both"):
        if b_gen_crops:
            sig_sims = compute_sims(b_gen_crops, b_gt_crops,
                                    model_siglip, proc_siglip, device, args.batch_size)
            result["gt_box_siglip_mean"] = float(sig_sims.mean())
            result["gt_box_siglip_std"] = float(sig_sims.std())
            result["gt_box_num_pairs"] = len(b_gen_crops)
        else:
            result["gt_box_siglip_mean"] = 0.0
            result["gt_box_siglip_std"] = 0.0
            result["gt_box_num_pairs"] = 0

    result["num_frames"] = frame_id
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(output_dir, mode):
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

    summary = {"evaluated": len(all_metrics), "mode": mode}

    if mode in ("iou_gated", "both"):
        for metric, label in [
            ("iou_gated_siglip_mean", "iou_gated_siglip"),
            ("detection_rate", "detection_rate"),
        ]:
            mu, sd, med = mean_std_med(metric)
            summary[f"{label}_mean"] = mu
            summary[f"{label}_std"] = sd
            summary[f"{label}_median"] = med

    if mode in ("gt_box", "both"):
        mu, sd, med = mean_std_med("gt_box_siglip_mean")
        summary["gt_box_siglip_mean"] = mu
        summary["gt_box_siglip_std"] = sd
        summary["gt_box_siglip_median"] = med

    summary_path = osp.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = osp.join(output_dir, "per_clip_metrics.csv")
    fieldnames = ["clip",
                  "iou_gated_siglip_mean", "detection_rate",
                  "gt_box_siglip_mean",
                  "iou_gated_num_pairs", "gt_box_num_pairs", "num_frames"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for m in sorted(all_metrics, key=lambda x: x["clip"]):
            w.writerow(m)

    logger.info("\n" + "=" * 60)
    logger.info("T7 FEATURE SIMILARITY (SigLIP2) SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Clips evaluated: {len(all_metrics)}")
    if "iou_gated_siglip_mean" in summary:
        logger.info("[Mode A IoU-gated]")
        logger.info(f"  SigLIP : {summary['iou_gated_siglip_mean']:.4f} +/- {summary['iou_gated_siglip_std']:.4f}")
        logger.info(f"  DetRate: {summary['detection_rate_mean']:.4f} +/- {summary['detection_rate_std']:.4f}")
    if "gt_box_siglip_mean" in summary:
        logger.info("[Mode B GT-box]")
        logger.info(f"  SigLIP : {summary['gt_box_siglip_mean']:.4f} +/- {summary['gt_box_siglip_std']:.4f}")
    logger.info(f"Summary: {summary_path}")
    logger.info(f"CSV:     {csv_path}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = make_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.aggregate_only:
        aggregate_results(args.output_dir, args.mode)
        return

    if args.gt_baseline:
        assert args.gt_list and args.sport, \
            "--gt_list and --sport required for --gt_baseline"
    else:
        assert args.results_dir and args.video_dir and args.gt_list and args.sport, \
            "--results_dir, --video_dir, --gt_list, --sport required unless --aggregate_only"

    logger.info(f"Mode: {args.mode}  Sport: {args.sport}  GT-baseline: {args.gt_baseline}")

    gt_mapping = build_gt_mapping(args.gt_list)
    logger.info(f"GT mapping: {len(gt_mapping)} entries")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading SigLIP2...")
    model_siglip, proc_siglip = load_siglip(args.siglip_model, device)
    logger.info("SigLIP2 loaded.")

    failed = []

    if args.gt_baseline:
        clip_entries = list(gt_mapping.items())
        if args.num_clips:
            clip_entries = clip_entries[:args.num_clips]
        logger.info(f"GT baseline: {len(clip_entries)} clips")

        for idx, (clip_name, gt_bbox_path) in enumerate(clip_entries):
            out_json = osp.join(args.output_dir, f"{clip_name}.json")
            if args.skip_existing and osp.exists(out_json):
                continue
            gt_video_path = bbox_path_to_video_path(gt_bbox_path, args.sport)
            if gt_video_path is None or not osp.exists(gt_video_path):
                logger.warning(f"[{idx+1}] GT video not found: {gt_video_path}")
                failed.append(clip_name)
                continue
            try:
                orig_gt_bboxes = load_bbox_file(gt_bbox_path)
                metrics = process_clip(
                    clip_name, gt_video_path, gt_video_path,
                    orig_gt_bboxes, orig_gt_bboxes, orig_gt_bboxes,
                    model_siglip, proc_siglip, device, args,
                )
                if metrics is None:
                    failed.append(clip_name)
                    continue
                with open(out_json, "w") as f:
                    json.dump(metrics, f, indent=2)
                if (idx + 1) % 20 == 0 or idx == len(clip_entries) - 1:
                    parts = [f"[{idx+1}/{len(clip_entries)}] {clip_name}"]
                    if "iou_gated_siglip_mean" in metrics:
                        parts.append(f"A:sig={metrics['iou_gated_siglip_mean']:.4f} "
                                     f"det={metrics['detection_rate']:.3f}")
                    if "gt_box_siglip_mean" in metrics:
                        parts.append(f"B:sig={metrics['gt_box_siglip_mean']:.4f}")
                    logger.info("  ".join(parts))
            except Exception as e:
                logger.error(f"[{idx+1}] ERROR {clip_name}: {e}")
                import traceback; traceback.print_exc()
                failed.append(clip_name)

        logger.info(f"Done. Failed: {len(failed)}")
        return

    # Regular mode: generated video vs GT video
    subdirs = sorted(glob.glob(osp.join(args.results_dir, "miou_results_*")))
    if subdirs:
        clip_dirs = []
        for sd in subdirs:
            for name in sorted(os.listdir(sd)):
                clip_dirs.append(osp.join(sd, name))
    else:
        clip_dirs = [osp.join(args.results_dir, c)
                     for c in sorted(os.listdir(args.results_dir))
                     if osp.isdir(osp.join(args.results_dir, c))]

    logger.info(f"Found {len(clip_dirs)} clip dirs")
    if args.num_clips:
        clip_dirs = clip_dirs[:args.num_clips]

    for idx, clip_dir in enumerate(clip_dirs):
        clip_name = osp.basename(clip_dir)
        out_json = osp.join(args.output_dir, f"{clip_name}.json")
        if args.skip_existing and osp.exists(out_json):
            continue

        pred_path = osp.join(clip_dir, "generated.txt")
        if not osp.isfile(pred_path):
            pred_path = osp.join(clip_dir, f"{clip_name}.txt")

        gt_trans_path = osp.join(clip_dir, "gt_bbox_transformed.txt")
        if not osp.isfile(gt_trans_path):
            gt_trans_path = osp.join(clip_dir, f"{clip_name}_gt_frame0only.txt")

        gen_video_path = osp.join(args.video_dir, clip_name, "generated.mp4")
        if not osp.isfile(gen_video_path):
            gen_video_path = osp.join(args.video_dir, f"{clip_name}.mp4")

        if not osp.isfile(pred_path) or not osp.isfile(gt_trans_path):
            logger.warning(f"[{idx+1}] Missing bbox files: {clip_name}")
            failed.append(clip_name)
            continue
        if not osp.isfile(gen_video_path):
            logger.warning(f"[{idx+1}] Missing generated video: {gen_video_path}")
            failed.append(clip_name)
            continue

        gt_bbox_path = gt_mapping.get(clip_name)
        if gt_bbox_path is None:
            logger.warning(f"[{idx+1}] Not in GT mapping: {clip_name}")
            failed.append(clip_name)
            continue
        gt_video_path = bbox_path_to_video_path(gt_bbox_path, args.sport)
        if gt_video_path is None or not osp.exists(gt_video_path):
            logger.warning(f"[{idx+1}] GT video not found: {gt_video_path}")
            failed.append(clip_name)
            continue

        try:
            pred_bboxes = load_bbox_file(pred_path)
            gt_bbox_transformed = load_bbox_file(gt_trans_path)
            orig_gt_bboxes = load_bbox_file(gt_bbox_path)

            metrics = process_clip(
                clip_name, gen_video_path, gt_video_path,
                gt_bbox_transformed, pred_bboxes, orig_gt_bboxes,
                model_siglip, proc_siglip, device, args,
            )
            if metrics is None:
                failed.append(clip_name)
                continue
            with open(out_json, "w") as f:
                json.dump(metrics, f, indent=2)
            if (idx + 1) % 20 == 0 or idx == len(clip_dirs) - 1:
                parts = [f"[{idx+1}/{len(clip_dirs)}] {clip_name}"]
                if "iou_gated_siglip_mean" in metrics:
                    parts.append(f"A:sig={metrics['iou_gated_siglip_mean']:.4f} "
                                 f"det={metrics['detection_rate']:.3f}")
                if "gt_box_siglip_mean" in metrics:
                    parts.append(f"B:sig={metrics['gt_box_siglip_mean']:.4f}")
                logger.info("  ".join(parts))
        except Exception as e:
            logger.error(f"[{idx+1}] ERROR {clip_name}: {e}")
            import traceback; traceback.print_exc()
            failed.append(clip_name)

    logger.info(f"Done. Failed: {len(failed)}")


if __name__ == "__main__":
    main()
