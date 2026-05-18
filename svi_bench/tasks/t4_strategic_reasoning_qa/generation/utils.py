import os
import json
import pandas as pd
import re
import random
from langdetect import detect

def get_language(s):
    try:
        return detect(s)
    except:
        return "Unknown"

def get_files(root):
    file_paths = []

    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))

    return file_paths

def get_asr_soccer(root):
    paths = get_files(root)
    asr = dict()

    for path in paths:
        with open(path, 'r') as f:
            l = json.load(f)

            for game_id in l:
                transcript = l[game_id]["ASR_transcript"]
                asr[game_id] = transcript

    return asr    

def get_asr_nhl(root):
    paths = get_files(root)
    asr = dict()

    for path in paths:
        with open(path, 'r') as f:
            l = json.load(f)
            
            for key in l:
                split = key.split("_period")
                game_id = split[0]
                period = int(split[1])
                transcript = l[key]["ASR_transcript"]

                if game_id not in asr:
                    asr[game_id] = []
                
                asr[game_id].append((period, transcript))

    return asr    

def get_asr(root, league, language=None):
    paths = get_files(root)
    asr = dict()

    not_language = set()

    for path in paths:
        if league in path:
            with open(path, 'r') as f:
                l = json.load(f)

                for item in l:
                    split = item["game_id"].split("_period")
                    game_id = split[0]
                    period = int(split[1])
                    transcript = item["ASR Transcript"]

                    if language != None:
                        if get_language(transcript) != language:
                            not_language.add(game_id)

                    if game_id not in asr:
                        asr[game_id] = []

                    asr[game_id].append((period, transcript))

    for id in not_language:
        del asr[id]
    
    return asr

def get_logs(root, game_ids):
    paths = get_files(root)
    logs = dict()

    for path in paths:
        game_id = path.split("/")[-1].split("_")[0]
        if game_id in game_ids:
            logs[game_id] = path
            
    return logs

def format_asr(items, segments=None, offset=60):
    items.sort()
    out = []

    for item in items:

        period, transcript = item

        out.append(f"\n**Period {period}**")

        for line in re.findall(r"\[(.*)\..*s -> (.*)\..*s\](.*)\n", transcript+"\n"):
            start = int(line[0].strip())
            end = int(line[1].strip())

            if segments != None:
                for segment in segments:
                    if period == segment[0] and start >= segment[1] - offset and end <= segment[2] + offset:
                        caption = line[2].strip()
                        if len(caption) > 3 and len(caption) <= 200:
                            out.append(f"[{start}-{end}] {caption}")
            else:
                caption = line[2].strip()
                if len(caption) > 3 and len(caption) <= 200:
                    out.append(f"[{start}-{end}] {caption}")

    return "\n".join(out)

def split_asr(items):
    if type(items) != list:
        items = [(i, items) for i in range(4)]

    items.sort()
    out = dict()

    for item in items:
        period, transcript = item

        lines = transcript.strip().split("\n")

        out[period] = []

        for line in lines:
            match = re.search(r"\[(.*)\..*s -> (.*)\..*s\](.*)", line)
            if match:
                start = int(match.group(1).strip())
                end = int(match.group(2).strip())
                caption = match.group(3).strip()
                out[period].append({
                    'start': start,
                    'end': end,
                    'caption': f"[{start}-{end}] {caption}"
                }) 
    
    return out

def get_paths(mapping_json, recap_directory, teams=None):
    with open(mapping_json, 'r') as f:
        l = json.load(f)

    paths = dict()
    
    for item in l:
        if teams != None:
            t = item.get("game_data", dict()).get("teams", "-").split("-")
            a, b = t[0].strip(), t[1].strip()
            if a not in teams or b not in teams:
                continue

        game_id = item.get("game_id_log") or item.get("game_id_fixture")
        log_path = item["source_paths"].get("log") or item["source_paths"].get("game_log")
        report_path = item["source_paths"]["espn"]

        recap_path = recap_directory + report_path.split("espn_report")[-1]

        if game_id and os.path.exists(log_path):
            paths[str(game_id)] = {
                "log": log_path,
                "report": recap_path if os.path.exists(recap_path) else None
            }

    return paths

def format_report(path):
    with open(path, 'r') as f:
        d = json.load(f)

    return f"**{d["title"]}**\n{d["recap"]}"

def format_nhl_logs(path):
    actions = {
        "Goals", "Shots", "Points", "Shots on goal",
        "Faceoffs Won", "Faceoffs Lost", "Power play shots", "Hits", "Entries", "Tackles"
        "Accurate passes", "Inaccurate passes", "Passes total", "Passes received","Foul", "Assists", 
        "Breakouts", "Puck battles", "Puck battles won", "Dump in", "Dump out", "Breakout successful",
        "Second assist", "Saves", "Giveaways", "Takeaways", "Puck recoveries"
    }

    columns = [
        "action_name", "player_name", "team_name", "opponent_name", "opponent_team_name",
        "half", "second", "zone_name", "opponent_zone_name", "possession_name", "possession_team_name",
        "possession_time", "possession_number", "attack_type_name", "second_clear_formatted",
        "shot_type", "shot_speed", "shot_distance", "goalie_view", "goalie_stance", "penalty_time", 
        "penalty_type", "penalty_violation"
    ]

    df = pd.read_csv(path, sep=";")

    df = df[columns]
    df = df[df['action_name'].isin(actions)]
    df = df.sort_values(by=["half", "second"])

    return df.to_csv(index=False, sep=",")

def extract_soccer_rosters(path):
    with open(path, 'r') as f:
        logs = json.load(f)

    rosters = dict()
    for event in logs["events"]:
        player = event.get("player", dict()).get("name", None)
        team = event.get("team", dict()).get("name", None)

        if player != None and team != None:
            if team not in rosters:
                rosters[team] = set()
            
            rosters[team].add(player)

    out = ""
    for team in rosters:
        players = list(rosters[team])
        out += f"{team}:\n"
        for player in players:
            out += f"{player}\n"
        out += "\n"
    
    return out.strip()

def extract_nhl_rosters(path):
    df = pd.read_csv(path, sep=";")

    teams = df['team_name'].dropna().unique().tolist()

    out = ""

    for team in teams:
        players = df[df['team_name'] == team]['player_name'].dropna().unique().tolist()
        out += f"{team}:\n"
        for player in players:
            out += f"{player}\n"
        out += "\n"

    return out.strip()

def extract_nba_rosters(path):
    actual_columns = [
        'id', 'action_id', 'action_name', 'player_id', 'player_name',
        'team_id', 'team_name', 'opponent_id', 'opponent_name',
        'opponent_team_id', 'opponent_team_name', 'teammate_id',
        'teammate_name', 'period', 'second', 'pos_x', 'pos_y',
        'possession_id', 'possession_name', 'possession_team_id',
        'possession_team_name', 'possession_number', 'possession_start_clear',
        'possession_end_clear', 'playtype', 'hand', 'shot_type',
        'drive', 'dribble_move', 'contesting', 'ts'
    ]

    df = pd.read_csv(path, skiprows=1, header=None, sep=";")
    df.columns = actual_columns

    teams = df['team_name'].dropna().unique().tolist()

    out = ""

    for team in teams:
        players = df[df['team_name'] == team]['player_name'].dropna().unique().tolist()
        out += f"{team}:\n"
        for player in players:
            out += f"{player}\n"
        out += "\n"

    return out.strip()

def format_nba_logs(path, target_period = None, pick_segment=False):
    actual_columns = [
        'id', 'action_id', 'action_name', 'player_id', 'player_name',
        'team_id', 'team_name', 'opponent_id', 'opponent_name',
        'opponent_team_id', 'opponent_team_name', 'teammate_id',
        'teammate_name', 'period', 'second', 'pos_x', 'pos_y',
        'possession_id', 'possession_name', 'possession_team_id',
        'possession_team_name', 'possession_number', 'possession_start_clear',
        'possession_end_clear', 'playtype', 'hand', 'shot_type',
        'drive', 'dribble_move', 'contesting', 'ts'
    ]

    actions = {
        "Screen", "2+", "2-", "Rebound", "Steal", "Block",
        "Foul", "Accurate pass", "Error leading to goal",
        "1+", "1-", "3+", "3-", "Pick'n'Roll", "Turnover",
        "Timeout", "2F", "3F", "2+1", "3+1", "Second chance",
        "Assisting", "Screen", "Post", "Technical foul"
    }

    columns = [
        "action_name", "player_name", "team_name", "opponent_name",
        "opponent_team_name", "teammate_name", "period", "second",
        "playtype", "hand", "shot_type", "drive", "dribble_move",
        "contesting"
    ]

    df = pd.read_csv(path, skiprows=1, header=None, sep=";")
    df.columns = actual_columns

    df = df[columns]
    df = df[df['action_name'].isin(actions)]
    
    df = df.sort_values(by=["period", "second"])

    if target_period:
        df = df[df['period'] == target_period]
        if pick_segment:
            period_end = df['second'].max()
            safe_df = df[df['second'] <= period_end - 60]
            start = safe_df['second'].sample().iloc[0]
            df = df[df['second'].between(start, start+60)]
            return (df.to_csv(index=False, sep=","), (start, start+60))

    return df.to_csv(index=False, sep=",")

def get_pool(dir):
    pool = set()
    for file in get_files(dir):
        with open(file, 'r') as f:
            for line in f:
                pool.add(line.strip())
    return pool

def parse_qa(response):
    questions = re.findall(r"Question [1-3]: (.*)", response)
    correct = re.findall(r"Correct answer: (.*)", response)
    answers = re.findall(r"Wrong answer [1-3]: (.*)", response)
    answers = [answers[i:i+3] for i in range(0, len(answers), 3)]

    return questions, correct, answers

def generate_play_by_play(log_path, league_code="3_NBA"):
    """Generate play-by-play log from game CSV. Returns None on error."""

    ACTION_NAME_MAP = {
        '1+': 'Free Throw Made',
        '1-': 'Free Throw Missed',
        '2+': '2 PT Made',
        '2+1': 'And One',
        '2-': '2 PT Missed',
        '3+': '3 PT Made',
        '3-': '3 PT Missed',
        '3+1': '3-Point And-One',
        'Assisting': 'Assisting',
        'Rebound': 'Rebound',
        'Turnover': 'Turnover',
        'Foul': 'Foul',
        'Technical foul': 'Technical Foul',
        "Pick'n'Roll": "Pick'n'Roll",
        'Screen': 'Screen/Post',
        'Post': 'Screen/Post',
        'Steal': 'Steal',
    }

    ALLOWED_RAW_ACTIONS = [
        '1+', '1-', '2+', '2+1', '2-', '3+', '3-', 
        'Assisting', 'Rebound', 'Turnover', 'Foul', 'Technical foul',
        "Pick'n'Roll", 'Screen', 'Post', 'Steal'
    ]

    actual_columns = [
        'id', 'action_id', 'action_name', 'player_id', 'player_name',
        'team_id', 'team_name', 'opponent_id', 'opponent_name',
        'opponent_team_id', 'opponent_team_name', 'teammate_id',
        'teammate_name', 'half', 'second', 'pos_x', 'pos_y',
        'possession_id', 'possession_name', 'possession_team_id',
        'possession_team_name', 'possession_number', 'possession_start_clear',
        'possession_end_clear', 'playtype', 'hand', 'shot_type',
        'drive', 'dribble_move', 'contesting', 'ts'
    ]

    if not os.path.exists(log_path):
        print(f"{log_path} does not exist")
        return None
    
    try:
        df = pd.read_csv(log_path, skiprows=1, header=None, sep=";")
    except Exception:
        print(f"{log_path} csv read error")
        return None
    
    if df.empty:
        print(f"{log_path} empty csv")
        return None
    
    df.columns = actual_columns
    
    # Get the last two columns for score (format: "{Team Name}_score")
    score_columns = df.columns.tolist()[-2:] if len(df.columns) >= 2 else []
    
    # Extract team names from column names (remove "_score" suffix)
    team_names = []
    if len(score_columns) == 2:
        for col in score_columns:
            if col.endswith('_score'):
                team_name = col[:-6]  # Remove "_score" suffix
                team_names.append(team_name)
            else:
                team_names.append(col)  # Fallback if format is different
    
    # Filter to only include allowed actions
    df = df[df['action_name'].isin(ALLOWED_RAW_ACTIONS)].copy()
    
    if df.empty:
        return None
    
    # Sort by period then time
    action_order = {
        'Foul': 1, 'Technical foul': 1, "Pick'n'Roll": 1, 'Post': 1, 'Screen': 1,
        'Assisting': 2,
        '1+': 4, '1-': 4, '2+': 4, '2-': 4, '3+': 4, '3-': 4
    }
    df['action_sort'] = df['action_name'].apply(lambda x: action_order.get(x, 3))
    sort_time_col = 'possession_end_clear' if 'possession_end_clear' in df.columns else 'second'
    df.sort_values(by=['half', sort_time_col, 'action_sort'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    play_by_play = []
    shot_actions = ['2+', '2+1', '2-', '3+', '3-', '3+1']
    made_shot_actions = ['1+', '2+', '2+1', '3+', '3+1']  # Only log score for made shots
    
    current_period = None
    
    for _, row in df.iterrows():
        if pd.isna(row.get('player_name')):
            continue
        
        action_name = str(row['action_name'])
        if action_name not in ALLOWED_RAW_ACTIONS:
            continue
        
        period = int(row.get('half', 1))
        
        if period != current_period:
            if current_period is not None:
                play_by_play.append("")
            play_by_play.append(f"=== Period {period} ===")
            current_period = period
        
        player_name = str(row['player_name'])
        team_name = str(row.get('team_name', '')) if pd.notna(row.get('team_name')) else ''
        
        # Get timestamp
        if pd.notna(row.get('possession_end_clear')):
            elapsed_seconds = int(row['possession_end_clear'])
        else:
            elapsed_seconds = int(row.get('second', 0))
        
        # Determine period length
        if league_code in ["3_NBA", "68_China. CBA", "137_United States. NBA G-League"]:
            period_length = 12 * 60
        elif league_code in ["6_NCAA Division I"]:
            period_length = 20 * 60
        else:
            period_length = 10 * 60
        
        time_remaining = max(0, period_length - elapsed_seconds)
        remaining_minutes = time_remaining // 60
        remaining_seconds = time_remaining % 60
        timestamp = f"[Q{period} {remaining_minutes}:{remaining_seconds:02d}]"
        
        display_action = ACTION_NAME_MAP.get(action_name, action_name)
        
        desc_parts = [timestamp]
        if team_name:
            desc_parts.append(f"[{team_name}]")
        
        # Get score from last two columns if available (only for made shots)
        score_text = None
        if action_name in made_shot_actions and len(score_columns) == 2 and len(team_names) == 2:
            score_col1 = score_columns[0]
            score_col2 = score_columns[1]
            if pd.notna(row.get(score_col1)) and pd.notna(row.get(score_col2)):
                try:
                    score1 = int(row[score_col1])
                    score2 = int(row[score_col2])
                    team1 = team_names[0]
                    team2 = team_names[1]
                    score_text = f"- Score: {team1} {score1}-{score2} {team2}"
                except (ValueError, TypeError):
                    pass
        
        if action_name == 'Assisting':
            if pd.notna(row.get('teammate_name')):
                teammate = str(row['teammate_name'])
                desc_parts.append(f"{player_name} assisted by {teammate}")
            else:
                continue
        elif action_name in shot_actions:
            desc_parts.append(f"{player_name} {display_action}")
            if pd.notna(row.get('pos_x')) and pd.notna(row.get('pos_y')):
                position = get_position(row['pos_x'], row['pos_y'], league_code)
                if position:
                    desc_parts.append(f"from {position}")
            if pd.notna(row.get('shot_type')):
                desc_parts.append(f"({row['shot_type']})")
            if score_text:
                desc_parts.append(score_text)
        elif action_name in ['1+', '1-']:
            desc_parts.append(f"{player_name} {display_action}")
            if score_text:
                desc_parts.append(score_text)
        else:
            desc_parts.append(f"{player_name} {display_action}")
        
        play_by_play.append(' '.join(desc_parts))
    
    return '\n'.join(play_by_play)

def get_video_path(game_id, league):
    path = None
    if league == "NBA" or league == "NCAA":
        path = f"/mnt/sun/shared/datasets/sports_dataset/basketball/full_game_video/{game_id}_full.mp4"
    
    if league == "EuroLeague":
        path = f"euroleague_games/{game_id}_full.mp4"
    
    if league == "NHL":
        path = f"/mnt/sun/shared/datasets/sports_dataset/hockey/full_game_video/{game_id}_full.mp4"
    
    if league == "LaLiga" or league == "Premier League":
        paths = get_files("/mnt/sun/shared/datasets/sports_dataset/soccer/full_game_video")
        for p in paths:
            if game_id in p:
                path = p
    
    if os.path.exists(path):
        return path
    else:
        return None