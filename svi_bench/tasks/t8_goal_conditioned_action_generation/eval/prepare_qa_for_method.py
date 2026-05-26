"""Prepare T8 goal-accuracy QA for a specific method's generated videos.

Given:
  - the anonymized master QA (one Q*.json per question type, with `id`,
    `video`, `start_bbox`, `conversations`), downloaded into
    $SVI_BENCH_DATA/T8/basketball/qa_test/ by svi_bench/tasks/t7_motion_conditioned_generation/scripts/download_t7_t8.sh
  - a flat directory of the method's generated 5-second clips, each named
    <anon_id>.mp4

this script:
  1. discovers which generated clips are present,
  2. filters each Q*.json to keep only entries whose underlying clip exists
     in the method's outputs,
  3. renders a red bounding-box overlay on frame 0 of each kept clip
     (coords come from start_bbox in the master QA),
  4. writes the filtered + path-rewritten Q*.json files into
     <output_dir>/qa_json/, with `video` pointing at the rendered overlays
     under <output_dir>/rendered_videos/.

The output is what test_llavaov.py consumes.
"""
import argparse
import glob
import json
import multiprocessing
import os
import os.path as osp
import subprocess
import traceback

from tqdm import tqdm

# Question/player suffix on QA `id`s. Strip these to recover the bare clip id.
SUFFIXES = (
    "_player0", "_player1", "_player2",
    "_contested_shot", "_contested",
    "_play_type", "_shot_type",
    "_dribble_move", "_shooting_hand", "_drive_direction",
    "_spatial_position",
)


def strip_qa_suffix(qa_id: str) -> str:
    for s in SUFFIXES:
        if qa_id.endswith(s):
            return qa_id[: -len(s)]
    return qa_id


def get_video_size(src_path: str) -> tuple[int, int]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", src_path],
        stderr=subprocess.DEVNULL,
    ).decode().strip()
    w, h = out.split(",")
    return int(w), int(h)


def render_one(args_tuple):
    gen_video_path, output_path, start_bbox = args_tuple
    try:
        if not osp.exists(gen_video_path):
            return output_path, False, f"source not found: {gen_video_path}"

        w, h = get_video_size(gen_video_path)
        if isinstance(start_bbox, dict):
            x1n, y1n = start_bbox["x1"], start_bbox["y1"]
            x2n, y2n = start_bbox["x2"], start_bbox["y2"]
        else:
            x1n, y1n, x2n, y2n = start_bbox

        px1 = int(x1n * w)
        py1 = int(y1n * h)
        bw  = max(1, int((x2n - x1n) * w))
        bh  = max(1, int((y2n - y1n) * h))

        os.makedirs(osp.dirname(output_path), exist_ok=True)
        vf = (
            f"drawbox=x={px1}:y={py1}:w={bw}:h={bh}"
            f":color=red@1.0:t=3:enable='eq(n\\,0)'"
        )
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", gen_video_path,
             "-vf", vf,
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
             "-an", output_path],
            capture_output=True,
        )
        if result.returncode != 0:
            return output_path, False, result.stderr.decode()
        return output_path, True, None
    except Exception:
        return output_path, False, traceback.format_exc()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True,
                        help="Flat dir of generated <anon_id>.mp4 files")
    parser.add_argument("--qa_dir", required=True,
                        help="Anonymized master QA dir "
                             "(e.g. $SVI_BENCH_DATA/T8/basketball/qa_test)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write rendered_videos/ and qa_json/")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip ffmpeg render if the overlay mp4 already exists")
    args = parser.parse_args()

    gen_videos = {}
    for f in os.listdir(args.video_dir):
        if f.endswith(".mp4"):
            gen_videos[osp.splitext(f)[0]] = osp.join(args.video_dir, f)
    print(f"Generated videos found: {len(gen_videos)}")

    qa_json_files = sorted(glob.glob(osp.join(args.qa_dir, "Q*.json")))
    if not qa_json_files:
        raise SystemExit(f"No Q*.json under {args.qa_dir}")
    print(f"QA JSON files: {len(qa_json_files)}")

    rendered_video_dir = osp.join(args.output_dir, "rendered_videos")
    qa_output_dir      = osp.join(args.output_dir, "qa_json")
    os.makedirs(rendered_video_dir, exist_ok=True)
    os.makedirs(qa_output_dir, exist_ok=True)

    render_tasks = {}   # output_path -> (gen_video_path, output_path, start_bbox)
    filtered_qa = {}    # qa_filename -> list of filtered entries

    for jf in qa_json_files:
        jf_name = osp.basename(jf)
        samples = json.load(open(jf))

        kept = []
        for sample in samples:
            clip_id = strip_qa_suffix(sample["id"])
            if clip_id not in gen_videos:
                continue
            gen_video_path = gen_videos[clip_id]
            rendered_path = osp.join(rendered_video_dir, sample["id"] + ".mp4")
            start_bbox = sample.get("start_bbox")
            if start_bbox is None:
                # nothing to draw; symlink later
                render_tasks[rendered_path] = (gen_video_path, rendered_path, None)
            else:
                render_tasks[rendered_path] = (gen_video_path, rendered_path, start_bbox)

            kept.append({
                "id":            sample["id"],
                "video":         rendered_path,
                "question_type": "",
                "conversations": sample["conversations"],
            })

        if kept:
            filtered_qa[jf_name] = kept
            print(f"  {jf_name}: {len(kept)}/{len(samples)} matched")

    total = sum(len(v) for v in filtered_qa.values())
    print(f"\nTotal render tasks: {len(render_tasks)}")
    print(f"Total kept QA pairs: {total}")

    # Split: real render vs. plain symlink
    render_work = []
    symlink_count = 0
    skipped = 0
    for out_path, (gen_path, _, bbox) in render_tasks.items():
        if args.skip_existing and osp.exists(out_path):
            skipped += 1
            continue
        if bbox is None:
            os.makedirs(osp.dirname(out_path), exist_ok=True)
            if not osp.exists(out_path):
                os.symlink(gen_path, out_path)
            symlink_count += 1
        else:
            render_work.append((gen_path, out_path, bbox))

    print(f"Rendering {len(render_work)} with overlay, {symlink_count} symlinked, {skipped} skipped")

    if render_work:
        ok = fail = 0
        with multiprocessing.Pool(args.workers) as pool:
            for out_path, success, msg in tqdm(
                pool.imap_unordered(render_one, render_work, chunksize=4),
                total=len(render_work), desc="Rendering",
            ):
                if success:
                    ok += 1
                else:
                    fail += 1
                    tqdm.write(f"  FAIL: {msg}")
        print(f"Rendered: ok={ok}  fail={fail}")

    for jf_name, kept in filtered_qa.items():
        out_path = osp.join(qa_output_dir, jf_name)
        with open(out_path, "w") as f:
            json.dump(kept, f, indent=2)
        print(f"  wrote {out_path} ({len(kept)})")

    print(f"\nDone. Method QA bundle at: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
