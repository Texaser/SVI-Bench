import os
import cv2


def condense_video_to_1h(video_path, output_path, target_duration_seconds=3600):
    """
    Condense video to 1 hour (or specified duration) by reducing FPS to speed up playback.

    Args:
        video_path: Path to input video file
        output_path: Path to save the condensed video file
        target_duration_seconds: Target duration in seconds (default: 3600 for 1 hour)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0:
        cap.release()
        raise ValueError("Video has no frames")

    current_duration = total_frames / original_fps
    new_fps = total_frames / target_duration_seconds

    print(f"Original video properties:")
    print(f"  Duration: {current_duration:.2f} seconds ({current_duration/60:.2f} minutes)")
    print(f"  FPS: {original_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")

    if current_duration <= target_duration_seconds:
        print(f"Warning: Video is already {current_duration/60:.2f} minutes (<= {target_duration_seconds/60:.2f} minutes).")
        print(f"Video will be sped up to {new_fps:.2f} FPS (from {original_fps:.2f} FPS).")
    else:
        speed_factor = current_duration / target_duration_seconds
        print(f"Condensing video by factor of {speed_factor:.2f}x")
        print(f"New FPS: {new_fps:.2f} (original: {original_fps:.2f})")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video writer for: {output_path}")

    print(f"\nProcessing video...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frame_count += 1

        if frame_count % 1000 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)...")

    cap.release()
    out.release()

    print(f"\nSuccessfully condensed video!")
    print(f"  Output saved to: {output_path}")
    print(f"  New duration: {target_duration_seconds} seconds ({target_duration_seconds/60:.2f} minutes)")
    print(f"  New FPS: {new_fps:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Condense a video to a target duration by adjusting FPS")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output video")
    parser.add_argument("--duration", type=int, default=3600, help="Target duration in seconds (default: 3600)")
    args = parser.parse_args()
    condense_video_to_1h(args.input, args.output, args.duration)
