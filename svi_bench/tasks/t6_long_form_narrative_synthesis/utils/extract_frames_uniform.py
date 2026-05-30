import os
import cv2


def extract_frames_uniform(video_path, output_dir, num_frames=500):
    """
    Extract frames uniformly from video and save as JPEGs at half resolution.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract (default: 500)
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames < num_frames:
        print(f"Warning: Video has only {total_frames} frames, but {num_frames} requested.")
        num_frames = total_frames

    if total_frames == 0:
        raise ValueError("Video has no frames")

    new_width = original_width // 2
    new_height = original_height // 2
    print(f"Original resolution: {original_width}x{original_height}")
    print(f"Resizing frames to: {new_width}x{new_height} (half resolution)")

    if num_frames == 1:
        frame_indices = [0]
    else:
        frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]

    print(f"Extracting {num_frames} frames from video with {total_frames} total frames...")

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue

        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        output_path = os.path.join(output_dir, f"frame_{idx + 1}.jpg")
        cv2.imwrite(output_path, frame_resized)

        if (idx + 1) % 50 == 0:
            print(f"Extracted {idx + 1}/{num_frames} frames...")

    cap.release()
    print(f"Successfully extracted {len(frame_indices)} frames to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames uniformly from a video")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output_dir", required=True, help="Directory to save frames")
    parser.add_argument("--num_frames", type=int, default=500, help="Number of frames to extract (default: 500)")
    args = parser.parse_args()
    extract_frames_uniform(args.input, args.output_dir, args.num_frames)
