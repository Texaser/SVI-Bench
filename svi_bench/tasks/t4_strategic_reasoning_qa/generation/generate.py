import random
import os
from gpt import *
from utils import *
from prompt_templates import *
from pydantic import BaseModel, Field

class Timestamp(BaseModel):
    period: int = Field(..., description="In-game period.")
    start: int = Field(..., description="Start timestamp.")
    end: int = Field(..., description="End timestamp.")    

class QuestionAnswerPair(BaseModel):
    reasoning: str = Field(..., description="Brief analysis of the subtitles and game report before forming the final question and answer.")
    question: str = Field(..., description="A difficult question about the video based on the subtitles and game report.")
    subtitle_evidence: list[Timestamp] = Field(..., description="Timestamps for all subtitle segments that directly support the answer. Do not omit relevant segments.")
    report_evidence: list[str] = Field(..., description="All verbatim quotes from the game report that directly support the answer. Quotes must be exact and complete.")
    answer: str = Field(..., description="A factual answer derived strictly from the evidence. Must be at most 50 words and contain no speculation.")
    
class Output(BaseModel):
    qa_pairs: list[QuestionAnswerPair] = Field(..., description="A list of five difficult, diverse question-answer pairs grounded strictly in the subtitles and game report.")

class NBA:
    def __init__(self):
        ASR_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/asr/0914_high_quality_league_asr"
        MAPPING_JSON = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/game_report/1007_NBA_espn_log_mapping.json"
        RECAP_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/game_report/NBA_espn_recaps"
        POOL_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_set_split/0917_game_id_split/pool1/basketball"

        self.asr = get_asr(ASR_DIR, "NBA")
        self.paths = get_paths(MAPPING_JSON, RECAP_DIR)
        self.pool = get_pool(POOL_DIR)
        self.games = []

        game_ids = self.asr.keys() & self.paths.keys() & self.pool
        for game_id in game_ids:
            if os.path.exists(f"/mnt/sun/shared/datasets/sports_dataset/basketball/full_game_video/{game_id}_full.mp4"):
                self.games.append(game_id)

        random.shuffle(self.games)

    def get_games(self):
        return self.games
    
    def get_asr(self, game_id, format=True):
        if format:
            return format_asr(self.asr[game_id])
        else:
            return self.asr[game_id]
    
    def get_rosters(self, game_id):
        return extract_nba_rosters(self.paths[game_id]["log"])
    
    def get_report(self, game_id):
        if self.paths[game_id]["report"] != None:
            return format_report(self.paths[game_id]["report"])
        else:
            return "N/A"

class NCAA:
    def __init__(self):
        ASR_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/asr/0914_high_quality_league_asr"
        MAPPING_JSON = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/game_report/1003_NCAAM_espn_log_mapping.json"
        RECAP_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/game_report/0903_NCAAM_espn_recaps"
        POOL_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_set_split/0917_game_id_split/pool1/basketball"

        self.asr = get_asr(ASR_DIR, "6_NCAA Division I")
        self.paths = get_paths(MAPPING_JSON, RECAP_DIR)
        self.pool = get_pool(POOL_DIR)
        self.games = []

        game_ids = self.asr.keys() & self.paths.keys() & self.pool
        for game_id in game_ids:
            if os.path.exists(f"/mnt/sun/shared/datasets/sports_dataset/basketball/full_game_video/{game_id}_full.mp4"):
                self.games.append(game_id)
        
        random.shuffle(self.games)

    def get_games(self):
        return self.games
    
    def get_asr(self, game_id, format=True):
        if format:
            return format_asr(self.asr[game_id])
        else:
            return self.asr[game_id]
    
    def get_rosters(self, game_id):
        return extract_nba_rosters(self.paths[game_id]["log"])
    
    def get_report(self, game_id):
        if self.paths[game_id]["report"] != None:
            return format_report(self.paths[game_id]["report"])
        else:
            return "N/A"

class EuroLeague:
    def __init__(self):
        ASR_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/asr/0914_high_quality_league_asr"
        LOG_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/game_logs"

        self.asr = get_asr(ASR_DIR, "2_Euroleague")
        self.logs = get_logs(LOG_DIR, self.asr.keys())
        self.games = list(self.asr.keys() & self.logs.keys())

        random.shuffle(self.games)

    def get_games(self):
        return self.games
    
    def get_asr(self, game_id, format=True):
        if format:
            return format_asr(self.asr[game_id])
        else:
            return self.asr[game_id]
    
    def get_rosters(self, game_id):
        return extract_nba_rosters(self.logs[game_id])
    
    def get_report(self, game_id):
        return "N/A"

class NHL:
    def __init__(self):
        ASR_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/asr/1007_hockey_full_game_asr"
        MAPPING_JSON = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/game_report/1019_NHL_espn_fixture_mapping.json"
        RECAP_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/more_sports/hockey/game_report_extraction/NHL_espn_recaps"
        POOL_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_set_split/0917_game_id_split/pool1/hockey"

        self.asr = get_asr_nhl(ASR_DIR)
        self.paths = get_paths(MAPPING_JSON, RECAP_DIR)
        self.pool = get_pool(POOL_DIR)
        self.games = []

        game_ids = self.asr.keys() & self.paths.keys() & self.pool
        for game_id in game_ids:
            if os.path.exists(f"/mnt/sun/shared/datasets/sports_dataset/hockey/full_game_video/{game_id}_full.mp4"):
                self.games.append(game_id)

        random.shuffle(self.games)

    def get_games(self):
        return self.games
    
    def get_asr(self, game_id, format=True):
        if format:
            return format_asr(self.asr[game_id])
        else:
            return self.asr[game_id]
    
    def get_rosters(self, game_id):
        return extract_nhl_rosters(self.paths[game_id]["log"])
    
    def get_report(self, game_id):
        if self.paths[game_id]["report"] != None:
            return format_report(self.paths[game_id]["report"])
        else:
            return "N/A"
    
class EPL:
    def __init__(self):
        ASR_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/asr/0325_soccer_pool1_asr/EPL"
        MAPPING_JSON = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/game_report/1124_EPL_espn_fixture_mapping.json"
        RECAP_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/more_sports/soccer/game_report_extraction/EPL_espn_recaps/"
        POOL_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_set_split/0917_game_id_split/pool1/soccer"

        self.asr = get_asr_soccer(ASR_DIR)
        self.paths = get_paths(MAPPING_JSON, RECAP_DIR)
        self.pool = get_pool(POOL_DIR)
        self.games = list(self.asr.keys() & self.paths.keys() & self.pool)

        random.shuffle(self.games)

    def get_games(self):
        return self.games
    
    def get_asr(self, game_id, format=False):
        if format:
            return format_asr(self.asr[game_id])
        else:
            return self.asr[game_id]
    
    def get_rosters(self, game_id):
        return extract_soccer_rosters(self.paths[game_id]["log"])
    
    def get_report(self, game_id):
        if self.paths[game_id]["report"] != None:
            return format_report(self.paths[game_id]["report"])
        else:
            return "N/A"
    
class LaLiga:
    def __init__(self):
        ASR_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/asr/0325_soccer_pool1_asr/LaLiga"
        MAPPING_JSON = "/mnt/opr/yulupan/basketball_QA_dataset/video_captioning/game_report/1124_LaLiga_espn_fixture_mapping.json"
        RECAP_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/more_sports/soccer/game_report_extraction/0903_LaLiga_espn_recaps"
        POOL_DIR = "/mnt/opr/yulupan/basketball_QA_dataset/video_set_split/0917_game_id_split/pool1/soccer"

        self.asr = get_asr_soccer(ASR_DIR)
        self.paths = get_paths(MAPPING_JSON, RECAP_DIR)
        self.pool = get_pool(POOL_DIR)
        self.games = list(self.asr.keys() & self.paths.keys() & self.pool)

        random.shuffle(self.games)

    def get_games(self):
        return self.games
    
    def get_asr(self, game_id, format=False):
        if format:
            return format_asr(self.asr[game_id])
        else:
            return self.asr[game_id]
    
    def get_rosters(self, game_id):
        return extract_soccer_rosters(self.paths[game_id]["log"])
    
    def get_report(self, game_id):
        if self.paths[game_id]["report"] != None:
            return format_report(self.paths[game_id]["report"])
        else:
            return "N/A"

def generate_qa(model, prompts, helper, num_per_type=1, reasoning=None):
    qa = []
    messages = []
    
    cur = 0
    custom_id = 0

    game_ids = helper.get_games()

    for question_type, question_prompt in prompts.items():
        for i in range(num_per_type):
            game_id = game_ids[cur]

            formatted_asr = helper.get_asr(game_id)
            # formatted_logs = format_nba_logs(paths[game_id]["log"])
            #formatted_logs = format_nhl_logs(paths[game_id]["log"])
            formatted_rosters = helper.get_rosters(game_id)
            formatted_report = helper.get_report(game_id)
            # formatted_rosters = extract_nba_rosters(logs[game_id])
            # formatted_report = "N/A"
            
            formatted_prompt = PROMPT.format(question_prompt, formatted_rosters, formatted_asr, formatted_report)
            messages.append(batch_object(formatted_prompt, model, custom_id, system_prompt=SYSTEM_PROMPT, reasoning=reasoning, format=Output))

            qa.append({
                "game_id": game_id,
                "question_type": question_type
            })

            cur += 1
            cur %= len(game_ids)
            custom_id += 1

    return qa, messages

def parse_responses(qa_file, batch_files, helper, cost=False):
    with open(qa_file, 'r') as f:
        qa = json.load(f)

    new_qa = []

    results = []
    for batch_file in batch_files:
        results += parse_batch(batch_file, cost=cost)

    results.sort(key= lambda x: int(x[0].split("task-")[-1])) # will be ordered by custom id

    total_cost = 0

    for result in results:
        custom_id, response, cost = result[0], result[1], result[2]

        i = int(custom_id.split("-")[-1])

        try:
            item = json.loads(response)
        except:
            print(f"response {i} is bad")
            continue

        total_cost += cost


        subtitles = split_asr(helper.get_asr(qa[i]["game_id"], format=False))

        # qa[i].pop("inputs", None)

        for pair in item["qa_pairs"]:
            pair["subtitle_evidence"].sort(key=lambda x: (x["period"], x["start"], x["end"]))

            for timestamp in pair["subtitle_evidence"]:
                aggregated = ""

                if timestamp["period"] not in subtitles:
                    print(pair)
                    continue

                for subtitle in subtitles[timestamp["period"]]:
                    if subtitle["start"] >= timestamp["start"] and subtitle["end"] <= timestamp["end"]:
                        aggregated += subtitle["caption"] + "\n"

                timestamp["subtitles"] = aggregated

            new_qa.append(qa[i] | pair)
    
    with open("qa.json", 'w', encoding='utf-8') as f:
        json.dump(new_qa, f, indent=4, ensure_ascii=False)

    print(total_cost)

setup = {
    'tactical_strategic_analysis': prompt_tactical_strategic_analysis,
    'role_skill_assessment': prompt_role_skill_assessment,
    'causal_counterfactual_reasoning': prompt_causal_counterfactual_reasoning,
    'anomaly_novelty_detection': prompt_anomaly_novelty_detection,
    'spatiotemporal_relational_reasoning': prompt_spatiotemporal_relational_reasoning,
    'general': prompt_general
}