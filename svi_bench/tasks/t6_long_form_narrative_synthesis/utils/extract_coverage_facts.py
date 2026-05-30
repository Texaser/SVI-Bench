"""
Extract Coverage Facts from Ground-Truth Reports (OpenAI Batch API)

Extracts observable, fine-grained game facts from ground-truth reports
using GPT via the OpenAI Batch API. Supports basketball, hockey, and soccer
with sport-specific prompts. Processes Q1-Q5 for single-game and multi-game.

Usage:
  export OPENAI_API_KEY="sk-..."

  # Submit and monitor
  python extract_coverage_facts.py --data_dir data/basketball --sport basketball

  # Only single-game or multi-game
  python extract_coverage_facts.py --data_dir data/hockey --sport hockey --mode single

  # Monitor existing batches only
  python extract_coverage_facts.py --data_dir data/soccer --sport soccer --monitor
"""

import json
import os
import time
import argparse
import tempfile
from openai import OpenAI
from tqdm import tqdm


# ==========================================
# CONFIGURATION
# ==========================================
MODEL = "gpt-4o"
QUESTION_TYPES = ["Q1", "Q2", "Q3", "Q4", "Q5"]

INPUT_REPORT_FILE = "ground_truth_report.txt"
OUTPUT_FACTS_FILE = "coverage_facts.txt"

BATCH_POLL_INTERVAL = 60

MAX_OUTPUT_TOKENS_SINGLE = 8000
MAX_OUTPUT_TOKENS_MULTI = 10000


# ==========================================
# SPORT-SPECIFIC PROMPT TEMPLATES
# ==========================================
PROMPT_TEMPLATES = {
    "basketball": """
### TASK
You are trying to extract all observable, fine-grained game facts from a basketball game report. Extract only information that can be directly observed from watching the game itself. Your job is to break down every sentence and extract every stat and every discrete in-game event described.

The extracted items must strictly reflect information that is visually or audibly verifiable from the live game broadcast. Do not include any contextual information that cannot be observed from the game itself, including: interviews; commentary about season framing (e.g., "season opener," "home debut"); player career context (e.g., "rookie," "returning from injury"); transactions; historical or team background; or any narrative not grounded in the on-court action.

Stats include—but are not limited to—score, points, rebounds, assists, steals, fouls, turnovers, shooting makes/misses, and explicitly stated quantifiable performance outcomes.

Key events include—but are not limited to—scoring plays, lead changes, momentum plays, clutch shots, defensive events, blocks, steals, or any other on-court actions that would be observable from watching the game.

Each extracted fact must:
• Describe exactly one stat or one event.
• Be self-contained and require no outside context.
• Refer to all entities explicitly by name (never by pronoun).
• Be stated as a single sentence (with zero or at most one embedded clause).
• Reflect only what occurs within the game itself.

Example facts:
- Team A won the game.
- The game result was 100-95.
- Player A scored 30 points.
- Player B grabbed 10 rebounds.
- Player C made a layup with 2:12 remaining.
- Player D shot 8-for-15 from the field.
- Team A led 58-52 at halftime.
- Team B made 3-of-8 three-pointers.

FINAL OUTPUT FORMAT (STRICT — NO EXPLANATIONS, NO SCORES, NO EXTRA text):
<fact 1>
<fact 2>
...

GROUND-TRUTH REPORT:
{INSERT_REPORT_HERE}
""",

    "hockey": """
### TASK
You are trying to extract all observable, fine-grained game facts from a hockey game report. Extract only information that can be directly observed from watching the game itself. Your job is to break down every sentence and extract every stat and every discrete in-game event described.

The extracted items must strictly reflect information that is visually or audibly verifiable from the live game broadcast. Do not include any contextual information that cannot be observed from the game itself, including: interviews; commentary about season framing (e.g., "season opener," "home debut"); player career context (e.g., "rookie," "returning from injury"); transactions; historical or team background; or any narrative not grounded in the on-ice action.

Stats include—but are not limited to—goals, assists, saves, shots on goal, power play goals, penalty minutes, plus/minus, faceoff wins, hits, blocked shots, time on ice, and explicitly stated quantifiable performance outcomes.

Key events include—but are not limited to—scoring plays, assists, saves, goals, power plays, penalty kills, lead changes, momentum plays, clutch goals, defensive events, blocked shots, fights, penalties, overtime goals, shootout attempts, or any other on-ice actions that would be observable from watching the game.

Each extracted fact must:
• Describe exactly one stat or one event.
• Be self-contained and require no outside context.
• Refer to all entities explicitly by name (never by pronoun).
• Be stated as a single sentence (with zero or at most one embedded clause).
• Reflect only what occurs within the game itself.

Example facts:
- Team A won the game.
- The game result was 4-3.
- Player A scored 2 goals.
- Player B recorded 3 assists.
- Player C scored a power-play goal with 2:12 remaining.
- Goaltender D made 28 saves.
- Team A led 2-1 after the first period.
- Team B went 1-for-4 on the power play.
- Player E received a 2-minute penalty for tripping at 12:45 of the third period.

FINAL OUTPUT FORMAT (STRICT — NO EXPLANATIONS, NO SCORES, NO EXTRA text):
<fact 1>
<fact 2>
...

GROUND-TRUTH REPORT:
{INSERT_REPORT_HERE}
""",

    "soccer": """
### TASK
You are trying to extract all observable, fine-grained game facts from a soccer match report. Extract only information that can be directly observed from watching the match itself. Your job is to break down every sentence and extract every stat and every discrete in-game event described.

The extracted items must strictly reflect information that is visually or audibly verifiable from the live match broadcast. Do not include any contextual information that cannot be observed from the match itself, including: interviews; commentary about season framing (e.g., "season opener," "home debut"); player career context (e.g., "rookie," "returning from injury"); transactions; historical or team background; or any narrative not grounded in the on-pitch action.

Stats include—but are not limited to—goals, assists, shots, passes, duels, fouls, yellow cards, red cards, recoveries, interceptions, saves, offsides, corners, and explicitly stated quantifiable performance outcomes.

Key events include—but are not limited to—scoring plays, assists, saves, goals, penalties, free kicks, corner kicks, lead changes, momentum plays, clutch goals, defensive events, clearances, tackles, substitutions, bookings, injury stoppages, extra-time goals, penalty shootout attempts, or any other on-pitch actions that would be observable from watching the match.

Each extracted fact must:
• Describe exactly one stat or one event.
• Be self-contained and require no outside context.
• Refer to all entities explicitly by name (never by pronoun).
• Be stated as a single sentence (with zero or at most one embedded clause).
• Reflect only what occurs within the match itself.

Do not keep compound statements. Whenever a sentence contains multiple facts (player + multiple stats, action + location + score, etc.), split into the smallest independently verifiable units. Each unit should be testable as Supported or Contradicted.

Example:
-Input: "Player A scored 2 goals and had 1 assist."
Atomic statements:
-Player A scored 2 goals.
-Player A had 1 assist.

Example facts:
- Team A won the match.
- The final score was 3-1.
- Player A scored 2 goals.
- Player B recorded 1 assist.
- Player C scored from a free kick in the 78th minute.
- Goalkeeper D made 6 saves.
- Team A led 1-0 at halftime.
- Player E received a yellow card for a foul in the 65th minute.

FINAL OUTPUT FORMAT (STRICT — NO EXPLANATIONS, NO SCORES, NO EXTRA text):
<fact 1>
<fact 2>
...

GROUND-TRUTH REPORT:
{INSERT_REPORT_HERE}
""",
}

SYSTEM_INSTRUCTIONS = {
    "basketball": "You are an expert sports data analyst specializing in extracting verifiable game facts from basketball reports.",
    "hockey": "You are an expert sports data analyst specializing in extracting verifiable game facts from hockey reports.",
    "soccer": "You are an expert sports data analyst specializing in extracting verifiable game facts from soccer match reports.",
}


# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def read_file_content(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def output_exists(entry_path: str, output_file: str) -> bool:
    out_path = os.path.join(entry_path, output_file)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def get_all_entries(base_dir: str) -> list:
    entries = []
    if not os.path.exists(base_dir):
        return entries
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            try:
                int(item)
                entries.append(item_path)
            except ValueError:
                continue
    entries.sort(key=lambda x: int(os.path.basename(x)))
    return entries


def extract_text_from_responses_body(body):
    """Robust extraction from batch response body."""
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            return ""

    if not isinstance(body, dict):
        return ""

    ot = body.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    out = body.get("output", [])
    if isinstance(out, list):
        texts = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        t = block.get("text")
                        if t:
                            texts.append(t)
            msg = item.get("message")
            if isinstance(msg, dict):
                content2 = msg.get("content")
                if isinstance(content2, list):
                    for block in content2:
                        if isinstance(block, dict):
                            t = block.get("text")
                            if t:
                                texts.append(t)
                elif isinstance(content2, str):
                    texts.append(content2)
        return "".join(texts).strip()
    return ""


# ==========================================
# TRACKING
# ==========================================
def load_tracking_data(tracking_file: str) -> dict:
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_tracking_data(data: dict, tracking_file: str):
    with open(tracking_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def update_batch_status(tracking_file, batch_id, status, output_file_id=None, error_file_id=None):
    data = load_tracking_data(tracking_file)
    if batch_id in data:
        data[batch_id]["status"] = status
        if output_file_id:
            data[batch_id]["output_file_id"] = output_file_id
        if error_file_id:
            data[batch_id]["error_file_id"] = error_file_id
        data[batch_id]["last_updated"] = time.time()
        save_tracking_data(data, tracking_file)


# ==========================================
# BATCH PREPARATION
# ==========================================
def prepare_batch_requests(label: str, entries: list, is_multi: bool,
                           input_file: str, output_file: str,
                           model: str, sport: str) -> tuple:
    requests = []
    mapping = {}

    max_tokens = MAX_OUTPUT_TOKENS_MULTI if is_multi else MAX_OUTPUT_TOKENS_SINGLE
    prompt_template = PROMPT_TEMPLATES[sport]
    system_instruction = SYSTEM_INSTRUCTIONS[sport]

    for entry_path in tqdm(entries, desc=f"Preparing {label}"):
        entry_name = os.path.basename(entry_path)
        custom_id = f"{label}_{entry_name}"

        report_path = os.path.join(entry_path, input_file)
        report_content = read_file_content(report_path)

        if not report_content:
            # Fallback to pseudo_ground_truth_report.txt
            fallback_path = os.path.join(entry_path, "pseudo_ground_truth_report.txt")
            report_content = read_file_content(fallback_path)
            if not report_content:
                continue

        if output_exists(entry_path, output_file):
            continue

        full_prompt = prompt_template.replace("{INSERT_REPORT_HERE}", report_content)

        req_body = {
            "model": model,
            "instructions": system_instruction,
            "input": full_prompt,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": "low"},
        }

        batch_req = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": req_body,
        }

        requests.append(batch_req)
        mapping[custom_id] = (entry_path, entry_name)

    return requests, mapping


# ==========================================
# SUBMISSION
# ==========================================
def submit_batches(client: OpenAI, data_dir: str, tracking_file: str,
                   mode: str, input_file: str, output_file: str,
                   model: str, sport: str):
    print(f"\n=== PHASE 1: SUBMISSION ({sport.upper()}) ===")
    tracking = load_tracking_data(tracking_file)

    game_types = []
    if mode in ("all", "single"):
        game_types.append(("single_game", False))
    if mode in ("all", "multi"):
        game_types.append(("multi_game", True))

    for game_type, is_multi in game_types:
        for q_type in QUESTION_TYPES:
            label = f"{game_type}_{q_type}"
            q_dir = os.path.join(data_dir, game_type, q_type)
            if not os.path.isdir(q_dir):
                continue

            active = [b for b, i in tracking.items()
                      if i.get("label") == label
                      and i["status"] not in ["completed", "failed", "cancelled", "expired"]]
            if active:
                print(f"[{label}] Batch active {active[0]}. Skipping.")
                continue

            all_entries = get_all_entries(q_dir)
            requests, mapping = prepare_batch_requests(
                label, all_entries, is_multi, input_file, output_file, model, sport)

            if not requests:
                print(f"[{label}] No requests to submit.")
                continue

            print(f"[{label}] Submitting {len(requests)} requests...")

            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
                for r in requests:
                    f.write(json.dumps(r) + "\n")
                temp_path = f.name

            try:
                with open(temp_path, "rb") as f:
                    file_obj = client.files.create(file=f, purpose="batch")

                batch_obj = client.batches.create(
                    input_file_id=file_obj.id,
                    endpoint="/v1/responses",
                    completion_window="24h",
                )

                tracking[batch_obj.id] = {
                    "label": label,
                    "is_multi": is_multi,
                    "status": "validating",
                    "entry_mapping": mapping,
                    "created_at": time.time(),
                    "output_file_id": None,
                    "error_file_id": None,
                }
                save_tracking_data(tracking, tracking_file)
                print(f"[{label}] Submitted batch {batch_obj.id}")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)


# ==========================================
# MONITORING & SAVING
# ==========================================
def download_text(client, file_id):
    try:
        return client.files.content(file_id).text
    except Exception as e:
        print(f"Error downloading {file_id}: {e}")
        return ""


def parse_jsonl(text):
    out = []
    for line in text.splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def save_results(results, mapping, errors_dir, output_file):
    count = 0
    for res in results:
        cid = res.get("custom_id")
        if cid not in mapping:
            continue

        entry_path, _ = mapping[cid]
        output_path = os.path.join(entry_path, output_file)

        if res.get("response", {}).get("status_code") == 200:
            body = res["response"]["body"]
            text = extract_text_from_responses_body(body)
            if text:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                count += 1
            else:
                with open(os.path.join(errors_dir, f"{cid}_empty.json"), "w") as f:
                    json.dump(res, f, indent=2)
    return count


def monitor_batches(client: OpenAI, tracking_file: str, data_dir: str, output_file: str):
    print("\n=== PHASE 2: MONITORING ===")

    errors_dir = os.path.join(data_dir, "batch_errors")
    outputs_dir = os.path.join(data_dir, "batch_outputs")
    os.makedirs(errors_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    while True:
        tracking = load_tracking_data(tracking_file)
        pending = [b for b, i in tracking.items()
                   if i["status"] not in ["completed", "failed", "cancelled", "expired"]]

        if not pending:
            print("No pending batches.")
            break

        print(f"\nChecking {len(pending)} batches at {time.strftime('%H:%M:%S')}...")

        for bid in pending:
            info = tracking[bid]
            label = info.get("label", "unknown")

            try:
                batch = client.batches.retrieve(bid)
                status = batch.status
                out_id = batch.output_file_id
                err_id = batch.error_file_id
                counts = batch.request_counts

                print(f"  [{label}] {bid[:12]}... : {status} (completed={counts.completed}/{counts.total})")

                if status == "completed":
                    if out_id:
                        print(f"    Downloading output {out_id}")
                        text = download_text(client, out_id)

                        with open(os.path.join(outputs_dir, f"{bid}_out.jsonl"), "w") as f:
                            f.write(text)

                        results = parse_jsonl(text)
                        n = save_results(results, info["entry_mapping"], errors_dir, output_file)
                        print(f"    Saved {n} results")

                    if err_id:
                        print(f"    Downloading errors {err_id}")
                        text = download_text(client, err_id)
                        with open(os.path.join(errors_dir, f"{bid}_err.jsonl"), "w") as f:
                            f.write(text)

                    update_batch_status(tracking_file, bid, "completed", out_id, err_id)

                elif status in ["failed", "cancelled", "expired"]:
                    update_batch_status(tracking_file, bid, status, out_id, err_id)
                else:
                    update_batch_status(tracking_file, bid, status)

            except Exception as e:
                print(f"Error checking {bid}: {e}")

        time.sleep(BATCH_POLL_INTERVAL)


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract coverage facts from ground-truth reports via OpenAI Batch API")
    parser.add_argument("--data_dir", required=True,
                        help="Path to sport data dir (e.g., data/basketball)")
    parser.add_argument("--sport", required=True, choices=["basketball", "hockey", "soccer"],
                        help="Sport type (determines prompt and token limits)")
    parser.add_argument("--model", default=MODEL,
                        help=f"GPT model name (default: {MODEL})")
    parser.add_argument("--mode", choices=["all", "single", "multi"], default="all",
                        help="Which game types to process: all, single, or multi")
    parser.add_argument("--monitor", action="store_true",
                        help="Only monitor existing batches")
    parser.add_argument("--input_file", default=INPUT_REPORT_FILE,
                        help=f"Input report filename (default: {INPUT_REPORT_FILE})")
    parser.add_argument("--output_file", default=OUTPUT_FACTS_FILE,
                        help=f"Output facts filename (default: {OUTPUT_FACTS_FILE})")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required.")
    client = OpenAI(api_key=api_key)

    tracking_file = os.path.join(args.data_dir, "batch_tracking.json")

    if not args.monitor:
        submit_batches(client, args.data_dir, tracking_file,
                       args.mode, args.input_file, args.output_file,
                       args.model, args.sport)

    monitor_batches(client, tracking_file, args.data_dir, args.output_file)


if __name__ == "__main__":
    main()
