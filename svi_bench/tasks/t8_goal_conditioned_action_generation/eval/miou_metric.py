"""
Extract bounding boxes from a single video using MixSort.
Supports video preprocessing (trimming/resampling) and tracking.
"""

import argparse
import os
import os.path as osp
import time
import cv2
import torch
import sys
import numpy as np
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), 'MixViT'))

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# try:
#     from decord import VideoWriter as DecordVideoWriter
#     HAVE_DECORD = True
# except ImportError:
#     DecordVideoWriter = None
#     HAVE_DECORD = False

try:
    import imageio_ffmpeg
    HAVE_IMAGEIO_FFMPEG = True
except Exception:
    imageio_ffmpeg = None
    HAVE_IMAGEIO_FFMPEG = False

try:
    import imageio
    HAVE_IMAGEIO = True
except Exception:
    imageio = None
    HAVE_IMAGEIO = False


def make_parser():
    parser = argparse.ArgumentParser("Extract bounding boxes from video using MixSort")
    
    # Input/Output
    parser.add_argument("--path", required=True, help="path to input video file")
    parser.add_argument("--output_dir", type=str, default="./mixsort_output", help="directory to save outputs")
    parser.add_argument("--save_result", action="store_true", default=True, help="save bbox tracks to txt file")
    parser.add_argument("--save_video", action="store_true", help="save annotated video with bboxes")
    parser.add_argument("--skip_existing", action="store_true", help="skip if output already exists")
    
    # Model config
    parser.add_argument("-f", "--exp_file", required=True, type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", required=True, type=str, help="checkpoint path")
    parser.add_argument("--device", default="gpu", type=str, help="cpu or gpu")
    parser.add_argument("--fp16", action="store_true", help="use half precision")
    parser.add_argument("--fuse", action="store_true", help="fuse conv+bn for testing")
    
    # Detection params
    parser.add_argument("--conf", default=0.1, type=float, help="detection confidence threshold")
    parser.add_argument("--nms", default=0.7, type=float, help="NMS IoU threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test image size")
    
    # Tracking params
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="frames to keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="filter boxes with high aspect ratio")
    parser.add_argument("--min_box_area", type=float, default=10, help="minimum box area to keep")
    
    # MixSort specific
    parser.add_argument("--script", type=str, default='mixformer_deit', help="MixFormer script name")
    parser.add_argument("--config", type=str, default='track', help="MixFormer config name")
    parser.add_argument("--alpha", type=float, default=0.6, help="fusion weight for IoU and appearance")
    parser.add_argument("--radius", type=int, default=0, help="radius for computing similarity")
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="IoU threshold for template update")
    parser.add_argument("--iou_only", action="store_true", help="use only IoU (no appearance)")
    parser.add_argument("--local_rank", type=int, default=0, help="GPU device rank")
    
    # Video preprocessing
    parser.add_argument("--clip_start_sec", type=float, default=None, help="start time in seconds (None=from beginning)")
    parser.add_argument("--clip_end_sec", type=float, default=None, help="end time in seconds (None=until end)")
    parser.add_argument("--clip_fps", type=float, default=None, help="resample to this fps (None=keep original)")
    parser.add_argument("--fps", default=30, type=int, help="fallback fps if video fps cannot be detected")
    
    # Debug/Analysis
    parser.add_argument("--save_first_iou", action="store_true", help="save first-frame detection IoU matrix")
    parser.add_argument("--save_per_frame_iou", action="store_true", help="save IoU matrix for every frame")
    parser.add_argument("--verbosity", type=int, default=20, help="log every N frames")
    
    # mIoU Metric Calculation
    parser.add_argument("--gt_bbox", type=str, default=None, help="path to ground truth bbox file (same format as output)")
    parser.add_argument("--compute_miou", action="store_true", help="compute mIoU metric against ground truth")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for matching (default: 0.5)")
    parser.add_argument("--save_miou_details", action="store_true", help="save per-frame IoU details")
    parser.add_argument("--penalize_unmatched", action="store_true", default=True,
                       help="penalize unmatched boxes by setting their IoU to 0 (default: True, recommended for video generation quality assessment)")
    parser.add_argument("--no_penalize_unmatched", dest="penalize_unmatched", action="store_false",
                       help="disable penalty for unmatched boxes (use standard mIoU that only considers matched boxes)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="only evaluate the first N frames (default: all frames)")

    # Coordinate transform for center-crop resizing
    parser.add_argument("--orig_video_width", type=int, default=None,
                       help="original video width before VAE preprocessing (e.g. 1280). "
                            "If set, GT bboxes will be transformed from original coordinate space "
                            "to the target video space using center-crop logic.")
    parser.add_argument("--orig_video_height", type=int, default=None,
                       help="original video height before VAE preprocessing (e.g. 720)")

    return parser


def compute_center_crop_transform(orig_w, orig_h, target_w, target_h):
    """
    Compute the coordinate transform parameters for center-crop resizing.

    DiffSynth-Studio's crop_and_resize:
      1. scale = max(target_w/orig_w, target_h/orig_h)
      2. Resize to (round(orig_w*scale), round(orig_h*scale))
      3. Center crop to (target_w, target_h)

    Returns (crop_left, crop_top, intermediate_w, intermediate_h, scale)
    """
    scale = max(target_w / orig_w, target_h / orig_h)
    intermediate_w = round(orig_w * scale)
    intermediate_h = round(orig_h * scale)
    crop_left = (intermediate_w - target_w) / 2.0
    crop_top = (intermediate_h - target_h) / 2.0
    return crop_left, crop_top, intermediate_w, intermediate_h, scale


def transform_bbox_center_crop(nx, ny, orig_w, orig_h, target_w, target_h, scale, crop_left, crop_top):
    """
    Transform a normalized coordinate from original video space to target (center-cropped) video space.

    nx, ny: normalized coordinates [0,1] in original video
    Returns: (nx_new, ny_new) normalized coordinates in target video
    """
    # Original pixel -> scaled pixel -> cropped pixel -> normalized in target
    px = nx * orig_w * scale - crop_left
    py = ny * orig_h * scale - crop_top
    return px / target_w, py / target_h


def transform_gt_bboxes(gt_bboxes_by_frame, orig_w, orig_h, target_w, target_h):
    """
    Transform all GT bboxes from original video coordinate space to target video space.
    Modifies bboxes in-place.

    Each bbox is [nx1, ny1, nx2, ny2, track_id, ...].
    """
    crop_left, crop_top, _, _, scale = compute_center_crop_transform(
        orig_w, orig_h, target_w, target_h
    )
    logger.info(f"Applying center-crop transform: orig={orig_w}x{orig_h} -> target={target_w}x{target_h}, "
                f"scale={scale:.4f}, crop_left={crop_left:.1f}, crop_top={crop_top:.1f}")

    for fid in gt_bboxes_by_frame:
        for box in gt_bboxes_by_frame[fid]:
            nx1, ny1, nx2, ny2 = box[0], box[1], box[2], box[3]
            box[0], box[1] = transform_bbox_center_crop(nx1, ny1, orig_w, orig_h, target_w, target_h, scale, crop_left, crop_top)
            box[2], box[3] = transform_bbox_center_crop(nx2, ny2, orig_w, orig_h, target_w, target_h, scale, crop_left, crop_top)

    return gt_bboxes_by_frame


def prepare_temporal_clip(args, output_path):
    """
    Preprocess video: trim to time range and/or resample to target fps.
    If no preprocessing needed, returns the original video path.
    """
    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {args.path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    clip_start_sec = args.clip_start_sec if args.clip_start_sec is not None else 0.0
    clip_end_sec = args.clip_end_sec
    target_fps = args.clip_fps if args.clip_fps and args.clip_fps > 0 else src_fps
    
    # Check if preprocessing is needed
    needs_preprocessing = False
    if clip_start_sec > 0.0:
        needs_preprocessing = True
    if clip_end_sec is not None:
        needs_preprocessing = True
    if args.clip_fps and abs(args.clip_fps - src_fps) > 0.01:
        needs_preprocessing = True
    
    # If no preprocessing needed, use original video
    if not needs_preprocessing:
        cap.release()
        logger.info(f"No preprocessing needed, using original video: {args.path}")
        return args.path, {
            "fps": src_fps,
            "frame_count": total_src_frames,
            "width": width,
            "height": height,
            "src_fps": src_fps,
        }
    
    # Otherwise, create preprocessed clip
    clip_path = output_path
    os.makedirs(osp.dirname(clip_path), exist_ok=True)
    if os.path.exists(clip_path):
        os.remove(clip_path)

    # Compute precise frame indices
    start_index_f = (clip_start_sec * src_fps) if src_fps > 0 else 0.0
    start_index = int(max(0, round(start_index_f)))
    
    if clip_end_sec is not None and clip_end_sec > clip_start_sec and src_fps > 0:
        end_index = int(max(start_index, round(clip_end_sec * src_fps)))
    else:
        end_index = total_src_frames - 1 if total_src_frames > 0 else start_index
    
    if end_index < start_index:
        end_index = start_index
    
    duration_sec = (end_index - start_index) / (src_fps if src_fps > 0 else 1.0)
    num_out = int(round(duration_sec * (target_fps if target_fps > 0 else src_fps)))
    if num_out <= 0:
        num_out = max(1, int((clip_end_sec - clip_start_sec) * (target_fps if target_fps > 0 else 1.0))) if clip_end_sec else 1

    used_imageio_writer = False
    writer = None
    if HAVE_IMAGEIO:
        # Use imageio+ffmpeg to write H.264 yuv420p directly (VS Code friendly)
        try:
            writer = imageio.get_writer(
                clip_path,
                fps=(target_fps if target_fps > 0 else src_fps),
                quality=None,  # Don't use quality, use CRF instead
                ffmpeg_params=[
                    "-crf", "23",               # Constant Rate Factor (default quality)
                    "-preset", "medium",        # Encoding speed preset
                    "-pix_fmt", "yuv420p",      # Pixel format for compatibility
                    "-movflags", "+faststart",  # Enable streaming
                ],
            )
            used_imageio_writer = True
            logger.info("Using imageio-ffmpeg writer for preprocessed clip (H.264 yuv420p, CRF=23).")
        except Exception as exc:
            writer = None
            logger.warning("imageio writer init failed (%s), falling back to OpenCV.", str(exc))
    if writer is None:
        # Initialize OpenCV writer with fallback codecs
        fourcc_codes = ["mp4v", "avc1"]
        for code in fourcc_codes:
            fourcc = cv2.VideoWriter_fourcc(*code)
            writer = cv2.VideoWriter(clip_path, fourcc, target_fps if target_fps > 0 else src_fps, (width, height))
            if writer.isOpened():
                logger.info(f"Using OpenCV VideoWriter codec '{code}' for preprocessed clip.")
                break
            writer.release()
            writer = None
        if writer is None:
            cap.release()
            raise RuntimeError("Failed to initialize OpenCV VideoWriter for preprocessed clip.")

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    current_index = start_index - 1
    current_frame = None
    written = 0
    for k in range(num_out):
        target_index = int(round(start_index_f + k * (src_fps / (target_fps if target_fps > 0 else src_fps))))
        if target_index > end_index:
            break
        # Advance to target_index
        while current_index < target_index:
            ret, frame = cap.read()
            if not ret:
                break
            current_index += 1
            current_frame = frame
        if current_frame is None:
            break
        if used_imageio_writer:
            writer.append_data(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
        else:
            writer.write(current_frame)
        written += 1

    cap.release()
    if used_imageio_writer:
        writer.close()
    else:
        writer.release()

    if written == 0:
        if os.path.exists(clip_path):
            os.remove(clip_path)
        raise RuntimeError("Temporal downsampling produced an empty clip. Check clip timings.")

    # If we used OpenCV writer and imageio-ffmpeg is available, transcode for VS Code compatibility
    if not used_imageio_writer:
        try:
            if HAVE_IMAGEIO_FFMPEG:
                ff_bin = imageio_ffmpeg.get_ffmpeg_exe()
                h264_tmp = clip_path + ".h264.mp4"
                ff_env = os.environ.copy()
                ff_env.pop("LD_LIBRARY_PATH", None)
                transcode_cmd = [
                    ff_bin, "-y", "-i", clip_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    h264_tmp,
                ]
                logger.info("Transcoding preprocessed clip to H.264 for VS Code: %s", " ".join(transcode_cmd))
                subprocess.run(transcode_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ff_env)
                os.replace(h264_tmp, clip_path)
        except Exception as exc:
            logger.warning("Transcoding preprocessed clip to H.264 failed, leaving original MP4V: %s", str(exc))

    logger.info(
        f"Prepared temporal clip: {clip_path} "
        f"(fps={(target_fps if target_fps > 0 else src_fps):.2f}, frames={written}, start={clip_start_sec}s, "
        f"end={clip_end_sec if clip_end_sec is not None else 'end'}s)"
    )

    return clip_path, {
        "fps": target_fps if target_fps > 0 else src_fps,
        "frame_count": written,
        "width": width,
        "height": height,
        "src_fps": src_fps,
    }


def load_bbox_file(bbox_path):
    """
    Load bounding boxes from txt file.
    Format: frame_id,track_id,x1,y1,x2,y2,score,-1,-1,-1
    Returns: dict mapping frame_id -> list of [x1,y1,x2,y2,track_id]
    """
    bboxes_by_frame = {}
    with open(bbox_path, 'r') as f:
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


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1, y1, x2, y2].
    Coordinates should be normalized (0-1) or in same scale.
    """
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def establish_track_id_mapping(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Establish track_id mapping between pred and gt based on IoU matching.
    
    Args:
        pred_boxes: list of predicted boxes [x1,y1,x2,y2,track_id]
        gt_boxes: list of ground truth boxes [x1,y1,x2,y2,track_id]
        iou_threshold: minimum IoU to consider a match
    
    Returns:
        mapping: dict {pred_track_id: gt_track_id}
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {}
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pred_box, gt_box)
    
    # Greedy matching
    matches = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((iou_matrix[i, j], i, j))
    matches.sort(reverse=True)
    
    matched_pred = set()
    matched_gt = set()
    mapping = {}
    
    for iou_val, i, j in matches:
        if i not in matched_pred and j not in matched_gt:
            pred_track_id = int(pred_boxes[i][4])
            gt_track_id = int(gt_boxes[j][4])
            mapping[pred_track_id] = gt_track_id
            matched_pred.add(i)
            matched_gt.add(j)
    
    return mapping


def compute_frame_miou(pred_boxes, gt_boxes, track_id_mapping, penalize_unmatched=False):
    """
    Compute mIoU for a single frame using pre-established track_id mapping.

    Args:
        pred_boxes: list of predicted boxes [x1,y1,x2,y2,track_id]
        gt_boxes: list of ground truth boxes [x1,y1,x2,y2,track_id]
        track_id_mapping: dict {pred_track_id: gt_track_id}
        penalize_unmatched: if True, assign IoU=0 to unmatched boxes

    Returns:
        mean_iou, matched_ious, num_matched, num_pred, num_gt
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, [], 0, len(pred_boxes), len(gt_boxes)

    # Build index for quick lookup
    pred_by_id = {int(box[4]): box for box in pred_boxes}
    gt_by_id = {int(box[4]): box for box in gt_boxes}

    matched_ious = []

    # Match using the mapping
    for pred_id, gt_id in track_id_mapping.items():
        if pred_id in pred_by_id and gt_id in gt_by_id:
            pred_box = pred_by_id[pred_id]
            gt_box = gt_by_id[gt_id]
            iou = compute_iou(pred_box, gt_box)
            matched_ious.append(iou)

    # Apply penalty for unmatched boxes if requested
    if penalize_unmatched:
        num_unmatched_pred = len(pred_boxes) - len(matched_ious)
        num_unmatched_gt = len(gt_boxes) - len(matched_ious)
        all_ious = matched_ious + [0.0] * (num_unmatched_pred + num_unmatched_gt)
        mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    else:
        mean_iou = np.mean(matched_ious) if len(matched_ious) > 0 else 0.0

    return mean_iou, matched_ious, len(matched_ious), len(pred_boxes), len(gt_boxes)


def find_first_appearance_frames(gt_bboxes):
    """
    Find the first frame where each GT track_id appears.
    
    Args:
        gt_bboxes: dict mapping frame_id -> list of boxes [x1,y1,x2,y2,track_id]
    
    Returns:
        dict mapping track_id -> first_frame_id
    """
    first_appearance = {}
    for frame_id in sorted(gt_bboxes.keys()):
        for box in gt_bboxes[frame_id]:
            track_id = int(box[4])
            if track_id not in first_appearance:
                first_appearance[track_id] = frame_id
    return first_appearance


def compute_video_miou(pred_bbox_path, gt_bbox_path, iou_threshold=0.5, save_details_path=None,
                       penalize_unmatched=False, max_frames=None):
    """
    Compute mIoU metric for entire video using dynamic ID mapping.
    Establishes track_id mapping when each GT track_id first appears.

    Args:
        pred_bbox_path: path to predicted bbox file
        gt_bbox_path: path to ground truth bbox file
        iou_threshold: minimum IoU to consider a match for establishing ID mapping
        save_details_path: if provided, save per-frame details to this path
        penalize_unmatched: if True, assign IoU=0 to unmatched boxes (lowers mIoU for false positives/negatives)
        max_frames: if provided, only evaluate the first N frames

    Returns:
        metrics: dict with overall metrics
    """
    logger.info(f"Loading predicted bboxes from: {pred_bbox_path}")
    pred_bboxes = load_bbox_file(pred_bbox_path)
    
    logger.info(f"Loading ground truth bboxes from: {gt_bbox_path}")
    gt_bboxes = load_bbox_file(gt_bbox_path)
    
    # Get all frame IDs
    all_frames = sorted(set(list(pred_bboxes.keys()) + list(gt_bboxes.keys())))
    if max_frames is not None and max_frames > 0:
        all_frames = all_frames[:max_frames]
        logger.info(f"Limiting evaluation to first {max_frames} frames")
    
    # Find when each GT track_id first appears
    gt_first_appearance = find_first_appearance_frames(gt_bboxes)
    logger.info(f"Found {len(gt_first_appearance)} unique GT track_ids")
    logger.info(f"GT track_id first appearance frames: {dict(sorted(gt_first_appearance.items()))}")

    # Establish track_id mapping dynamically
    track_id_mapping = {}  # pred_track_id -> gt_track_id
    mapped_gt_ids = set()  # Track which GT IDs have been mapped

    logger.info("Using dynamic ID mapping mode: establishing mapping when each GT track_id first appears")

    frame_metrics = []
    total_iou = 0.0
    total_matched = 0
    total_pred = 0
    total_gt = 0
    
    for frame_id in all_frames:
        pred_boxes = pred_bboxes.get(frame_id, [])
        gt_boxes = gt_bboxes.get(frame_id, [])
        
        # Dynamic mapping: check if any new GT track_ids appear in this frame
        new_gt_ids = [int(box[4]) for box in gt_boxes
                     if gt_first_appearance.get(int(box[4])) == frame_id
                     and int(box[4]) not in mapped_gt_ids]

        if new_gt_ids:
            # Filter boxes for new GT IDs
            new_gt_boxes = [box for box in gt_boxes if int(box[4]) in new_gt_ids]

            # Filter pred boxes that are not yet mapped
            mapped_pred_ids = set(track_id_mapping.keys())
            unmapped_pred_boxes = [box for box in pred_boxes if int(box[4]) not in mapped_pred_ids]

            if unmapped_pred_boxes and new_gt_boxes:
                new_mapping = establish_track_id_mapping(
                    unmapped_pred_boxes,
                    new_gt_boxes,
                    iou_threshold
                )

                if new_mapping:
                    logger.info(f"Frame {frame_id}: Establishing new mappings for {len(new_mapping)} track_ids:")
                    for pred_id, gt_id in sorted(new_mapping.items()):
                        logger.info(f"  Pred track_id {pred_id} → GT track_id {gt_id}")
                        track_id_mapping[pred_id] = gt_id
                        mapped_gt_ids.add(gt_id)

        mean_iou, matched_ious, num_matched, num_pred, num_gt = compute_frame_miou(
            pred_boxes, gt_boxes, track_id_mapping, penalize_unmatched=penalize_unmatched
        )
        
        frame_metrics.append({
            'frame_id': frame_id,
            'mean_iou': mean_iou,
            'num_matched': num_matched,
            'num_pred': num_pred,
            'num_gt': num_gt,
            'matched_ious': matched_ious
        })
        
        total_iou += mean_iou
        total_matched += num_matched
        total_pred += num_pred
        total_gt += num_gt
    
    # Compute overall metrics
    video_miou = total_iou / len(all_frames) if len(all_frames) > 0 else 0.0
    precision = total_matched / total_pred if total_pred > 0 else 0.0
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Log final mapping summary
    logger.info(f"\nFinal ID mapping summary: {len(track_id_mapping)} mappings established")
    logger.info(f"Mapped GT IDs: {sorted(mapped_gt_ids)}")
    logger.info(f"Total GT IDs: {len(gt_first_appearance)}")
    unmapped_gt_ids = set(gt_first_appearance.keys()) - mapped_gt_ids
    if unmapped_gt_ids:
        logger.info(f"Unmapped GT IDs: {sorted(unmapped_gt_ids)}")

    metrics = {
        'video_miou': video_miou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_frames': len(all_frames),
        'total_matched': total_matched,
        'total_pred': total_pred,
        'total_gt': total_gt,
        'iou_threshold': iou_threshold,
        'num_id_mappings': len(track_id_mapping),
        'num_gt_ids': len(gt_first_appearance),
        'num_mapped_gt_ids': len(mapped_gt_ids),
        'penalize_unmatched': penalize_unmatched
    }
    
    # Save per-frame details if requested
    if save_details_path:
        with open(save_details_path, 'w') as f:
            f.write("frame_id,mean_iou,num_matched,num_pred,num_gt\n")
            for fm in frame_metrics:
                f.write(f"{fm['frame_id']},{fm['mean_iou']:.6f},{fm['num_matched']},{fm['num_pred']},{fm['num_gt']}\n")
        logger.info(f"Saved per-frame details to: {save_details_path}")
    
    return metrics


class Predictor(object):
    def __init__(self, model, exp, device, fp16=False):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        img_info["file_name"] = None
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img_in, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img_in = torch.from_numpy(img_in).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img_in = img_in.half()

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img_in)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


def imageflow_demo_allframes(predictor, vis_folder, args, exp, gt_bboxes_by_frame=None):
    if args.device.type != 'cuda':
        raise RuntimeError("MixSort demo requires GPU (cuda).")

    if args.iou_only:
        from yolox.mixsort_tracker.mixsort_iou_tracker import MIXTracker as MIXTrackerImpl
    else:
        from yolox.mixsort_tracker.mixsort_tracker import MIXTracker as MIXTrackerImpl

    process_path = getattr(args, "processed_path", args.path)
    cap = cv2.VideoCapture(process_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {process_path}")

    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap_width) if cap_width > 0 else getattr(args, "processed_width", None) or 0
    height = int(cap_height) if cap_height > 0 else getattr(args, "processed_height", None) or 0
    fps = cap_fps if cap_fps > 0 else getattr(args, "processed_fps", None) or args.fps

    video_output_dir = getattr(args, "video_output_dir", vis_folder)
    bbox_output_dir = getattr(args, "bbox_output_dir", vis_folder)
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(bbox_output_dir, exist_ok=True)
    source_basename = osp.splitext(osp.basename(args.original_path if hasattr(args, "original_path") else args.path))[0]
    
    # Use annotated_video_path if provided, otherwise default
    if hasattr(args, "annotated_video_path") and args.annotated_video_path:
        save_path = args.annotated_video_path
    else:
        save_path = osp.join(video_output_dir, f"{source_basename}_annotated.mp4")

    vid_writer = None
    use_imageio_writer = False
    if args.save_video:
        clip_fps = getattr(args, "processed_fps", None) or (args.clip_fps if args.clip_fps and args.clip_fps > 0 else fps)
        logger.info(f"Video clip save_path is {save_path} (fps={clip_fps})")
        writer_size = (int(width), int(height))
        # Prefer imageio writer if available
        if HAVE_IMAGEIO:
            try:
                vid_writer = imageio.get_writer(
                    save_path,
                    fps=clip_fps,
                    quality=None,  # Don't use quality, use CRF instead
                    ffmpeg_params=[
                        "-crf", "23",          # Constant Rate Factor (23=default quality)
                        "-preset", "medium",   # Encoding speed preset
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                    ],
                )
                use_imageio_writer = True
                logger.info("Using imageio-ffmpeg writer for annotated video (H.264 yuv420p, CRF=23).")
            except Exception as exc:
                vid_writer = None
                logger.warning("imageio writer init failed for annotated video (%s), falling back to OpenCV.", str(exc))
        if vid_writer is None:
            # Initialize pure OpenCV writer with codec fallback
            for code in ["mp4v", "avc1"]:
                fourcc = cv2.VideoWriter_fourcc(*code)
                tmp_writer = cv2.VideoWriter(save_path, fourcc, clip_fps, writer_size)
                if tmp_writer.isOpened():
                    logger.info(f"Using OpenCV VideoWriter with codec '{code}'.")
                    vid_writer = tmp_writer
                    break
                tmp_writer.release()
            if vid_writer is None:
                raise RuntimeError("Failed to initialize OpenCV VideoWriter for annotated video.")
    else:
        clip_fps = None

    tracker = MIXTrackerImpl(args)
    tracker.re_init(args)
    from yolox.tracking_utils.timer import Timer
    timer = Timer()
    frame_id = 0
    results = []
    first_iou_saved = False

    # Precompute which GT track_ids first appear in which frame
    gt_first_appearance = {}
    if gt_bboxes_by_frame:
        for fid in sorted(gt_bboxes_by_frame.keys()):
            for box in gt_bboxes_by_frame[fid]:
                tid = int(box[4])
                if tid not in gt_first_appearance:
                    gt_first_appearance[tid] = fid
    gt_injected_ids = set()  # track which GT ids have been injected

    max_frames = getattr(args, 'max_frames', None)

    while True:
        if max_frames is not None and frame_id >= max_frames:
            logger.info(f'Reached max_frames={max_frames}, stopping tracking.')
            break
        if frame_id % max(1, args.verbosity) == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)

            # Inject GT bboxes for newly appearing players.
            # Frame 0: replace ALL detections with GT.
            # Later frames: append GT bboxes for new players to YOLOX detections.
            if gt_bboxes_by_frame and frame_id in gt_bboxes_by_frame:
                new_gt_boxes = [box for box in gt_bboxes_by_frame[frame_id]
                                if gt_first_appearance.get(int(box[4])) == frame_id
                                and int(box[4]) not in gt_injected_ids]
                if new_gt_boxes:
                    img_h, img_w = img_info['height'], img_info['width']
                    scale = min(predictor.test_size[0] / float(img_h),
                                predictor.test_size[1] / float(img_w))
                    gt_dets = []
                    for box in new_gt_boxes:
                        nx1, ny1, nx2, ny2 = box[0], box[1], box[2], box[3]
                        x1 = nx1 * img_w * scale
                        y1 = ny1 * img_h * scale
                        x2 = nx2 * img_w * scale
                        y2 = ny2 * img_h * scale
                        gt_dets.append([x1, y1, x2, y2, 1.0, 1.0, 0])
                    gt_tensor = torch.tensor(gt_dets, dtype=torch.float32).to(args.device)
                    for box in new_gt_boxes:
                        gt_injected_ids.add(int(box[4]))

                    if frame_id == 0:
                        # Frame 0: replace all YOLOX detections with GT
                        outputs = [gt_tensor]
                    else:
                        # Later frames: append new-player GT bboxes to YOLOX detections
                        if outputs[0] is not None:
                            outputs = [torch.cat([outputs[0], gt_tensor], dim=0)]
                        else:
                            outputs = [gt_tensor]
                    logger.info(f"Frame {frame_id}: injected {len(gt_dets)} GT bboxes for new players")

            if outputs[0] is not None:
                outs = outputs[0]
                if torch.is_tensor(outs):
                    outs = outs.detach().cpu().numpy()

                bboxes = outs[:, :4].copy()
                img_h, img_w = img_info['height'], img_info['width']
                scale = min(predictor.test_size[0] / float(img_h), predictor.test_size[1] / float(img_w))
                bboxes /= scale

                if args.save_first_iou and not first_iou_saved:
                    from yolox.mixsort_tracker import matching as _mt
                    iou_mat = _mt.ious([x for x in bboxes], [x for x in bboxes])
                    iou_txt = osp.join(bbox_output_dir, f"{source_basename}_frame0_iou.txt")
                    np.savetxt(iou_txt, iou_mat, fmt="%.6f")
                    logger.info(f"Saved first-frame IOU to {iou_txt}")
                    first_iou_saved = True

                if args.save_per_frame_iou:
                    from yolox.mixsort_tracker import matching as _mt
                    iou_mat = _mt.ious([x for x in bboxes], [x for x in bboxes])
                    iou_txt = osp.join(bbox_output_dir, f"{source_basename}_frame{frame_id:04d}_iou.txt")
                    np.savetxt(iou_txt, iou_mat, fmt="%.6f")

                origin_img = torch.from_numpy(frame).to(args.device)
                origin_img = origin_img.permute(2, 0, 1)

                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, origin_img)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                img_w = float(img_info['width'])
                img_h = float(img_info['height'])
                for tlwh, tid, score in zip(online_tlwhs, online_ids, online_scores):
                    nx1 = tlwh[0] / img_w
                    ny1 = tlwh[1] / img_h
                    nx2 = (tlwh[0] + tlwh[2]) / img_w
                    ny2 = (tlwh[1] + tlwh[3]) / img_h
                    results.append(
                        f"{frame_id},{tid},{nx1:.6f},{ny1:.6f},{nx2:.6f},{ny2:.6f},{score:.6f},-1,-1,-1\n"
                    )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )

                # Overlay GT bboxes in green for visualization/debug
                if args.save_video and gt_bboxes_by_frame and frame_id in gt_bboxes_by_frame:
                    img_h_f, img_w_f = float(img_info['height']), float(img_info['width'])
                    for box in gt_bboxes_by_frame[frame_id]:
                        nx1, ny1, nx2, ny2 = box[0], box[1], box[2], box[3]
                        gt_tid = int(box[4])
                        gx1 = int(nx1 * img_w_f)
                        gy1 = int(ny1 * img_h_f)
                        gx2 = int(nx2 * img_w_f)
                        gy2 = int(ny2 * img_h_f)
                        cv2.rectangle(online_im, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                        cv2.putText(online_im, f"GT{gt_tid}", (gx1, gy2 + 15),
                                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_video and vid_writer is not None:
                if use_imageio_writer:
                    vid_writer.append_data(cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB))
                else:
                    vid_writer.write(online_im)  # OpenCV expects BGR

            try:
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            except cv2.error:
                pass
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(bbox_output_dir, f"{source_basename}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"Saved results to {res_file}")

    cap.release()
    if vid_writer is not None:
        if use_imageio_writer:
            vid_writer.close()
        else:
            vid_writer.release()
        # Transcode annotated MP4 to H.264 if possible (ensures VS Code preview)
        if not use_imageio_writer:
            try:
                if HAVE_IMAGEIO_FFMPEG:
                    ff_bin = imageio_ffmpeg.get_ffmpeg_exe()
                    h264_tmp = save_path + ".h264.mp4"
                    ff_env = os.environ.copy()
                    ff_env.pop("LD_LIBRARY_PATH", None)
                    transcode_cmd = [
                        ff_bin, "-y", "-i", save_path,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        h264_tmp,
                    ]
                    logger.info("Transcoding annotated video to H.264 for VS Code: %s", " ".join(transcode_cmd))
                    subprocess.run(transcode_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ff_env)
                    os.replace(h264_tmp, save_path)
            except Exception as exc:
                logger.warning("Transcoding annotated video to H.264 failed, leaving original MP4V: %s", str(exc))


def main(args):
    # Setup paths
    os.makedirs(args.output_dir, exist_ok=True)
    args.device = torch.device("cuda" if (args.device == "gpu" and torch.cuda.is_available()) else "cpu")
    args.original_path = args.path
    
    source_basename = osp.splitext(osp.basename(args.path))[0]
    logger.info(f"Processing video: {args.path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Define output paths
    preprocessed_video_path = osp.join(args.output_dir, f"{source_basename}_preprocessed.mp4")
    bbox_output_path = osp.join(args.output_dir, f"{source_basename}.txt")
    annotated_video_path = osp.join(args.output_dir, f"{source_basename}_annotated.mp4") if args.save_video else None
    
    # Skip if output already exists
    if args.skip_existing and os.path.exists(bbox_output_path):
        if not args.save_video or (annotated_video_path and os.path.exists(annotated_video_path)):
            logger.info(f"Skipping: output already exists at {bbox_output_path}")
            return
    
    # Preprocess video (trim/resample) if needed
    processed_path, clip_info = prepare_temporal_clip(args, preprocessed_video_path)
    # Store processed video info
    args.processed_path = processed_path
    args.processed_fps = clip_info["fps"]
    args.processed_width = clip_info["width"]
    args.processed_height = clip_info["height"]
    
    if processed_path != args.path:
        logger.info(
            f"Video preprocessed: {processed_path} "
            f"(fps={clip_info['fps']:.2f}, frames={clip_info['frame_count']}, {clip_info['width']}x{clip_info['height']})"
        )
    
    # Set output directories for imageflow_demo_allframes
    args.video_output_dir = args.output_dir
    args.bbox_output_dir = args.output_dir
    
    # Override save_path if annotated video requested
    if args.save_video and annotated_video_path:
        args.annotated_video_path = annotated_video_path
    
    # Load experiment config
    exp = get_exp(args.exp_file, None)
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    
    # Load model
    logger.info("Loading model...")
    model = exp.get_model().to(args.device)
    model.eval()
    
    if args.ckpt is None:
        raise ValueError("Checkpoint path required (--ckpt)")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    logger.info(f"Model loaded from {args.ckpt}")
    
    if args.fuse:
        logger.info("Fusing model...")
        model = fuse_model(model)
    
    if args.fp16:
        logger.info("Using FP16...")
        model = model.half()
    
    # Run tracking
    predictor = Predictor(model, exp, args.device, args.fp16)
    logger.info("Running MixSort tracking...")
    imageflow_demo_allframes(predictor, args.output_dir, args, exp)
    
    logger.info(f"✓ Done! Bounding boxes saved to: {bbox_output_path}")
    if args.save_video and annotated_video_path:
        logger.info(f"✓ Annotated video saved to: {annotated_video_path}")
    
    # Compute mIoU metric if ground truth is provided
    if args.compute_miou or args.gt_bbox:
        if not args.gt_bbox:
            logger.warning("--gt_bbox not provided, skipping mIoU computation")
        elif not os.path.exists(args.gt_bbox):
            logger.error(f"Ground truth file not found: {args.gt_bbox}")
        else:
            logger.info("\n" + "="*60)
            logger.info("Computing mIoU Metric")
            logger.info("="*60)
            
            miou_details_path = None
            if args.save_miou_details:
                miou_details_path = osp.join(args.output_dir, f"{source_basename}_miou_details.csv")
            
            metrics = compute_video_miou(
                bbox_output_path,
                args.gt_bbox,
                iou_threshold=args.iou_threshold,
                save_details_path=miou_details_path,
                penalize_unmatched=args.penalize_unmatched,
                max_frames=args.max_frames
            )
            
            # Print metrics
            logger.info("\n" + "-"*60)
            logger.info("mIoU Metrics:")
            logger.info("-"*60)
            logger.info(f"Video mIoU:       {metrics['video_miou']:.4f}")
            logger.info(f"Precision:        {metrics['precision']:.4f} ({metrics['total_matched']}/{metrics['total_pred']})")
            logger.info(f"Recall:           {metrics['recall']:.4f} ({metrics['total_matched']}/{metrics['total_gt']})")
            logger.info(f"F1 Score:         {metrics['f1_score']:.4f}")
            logger.info(f"Total Frames:     {metrics['total_frames']}")
            logger.info(f"IoU Threshold:    {metrics['iou_threshold']:.2f}")
            logger.info("-"*60 + "\n")
            
            # Save metrics to JSON
            import json
            metrics_path = osp.join(args.output_dir, f"{source_basename}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"✓ Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)