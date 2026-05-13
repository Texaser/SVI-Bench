from evaluation.models.model import Model

from google import genai
import subprocess
import os
import time

class Gemini(Model):

    def __init__(self, key):
        self.client = genai.Client(api_key=key)

    def name(self):
        return "Gemini-3.1-Pro"
    
    def process_question(self, prompt, video_path):
        formatted_video = self.format_video(video_path)

        file = self.client.files.upload(file=formatted_video)
        os.remove(formatted_video)

        while True:
            info = self.client.files.get(name=file.name)
            if info.state == genai.types.FileState.ACTIVE:
                break
            elif info.state == genai.types.FileState.FAILED:
                raise Exception(f"video upload failed")
            time.sleep(60)

        for i in range(10):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.1-pro-preview",
                    contents=[file, prompt],
                    config=genai.types.GenerateContentConfig(
                        media_resolution=genai.types.MediaResolution.MEDIA_RESOLUTION_LOW
                    )
                )
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    self.client.files.delete(name=file.name)
                    raise
                
                time.sleep(600)
        else:
            raise Exception("Failed after 10 attempts")

        self.client.files.delete(name=file.name)

        return response.text

    def format_video(self, path):
        input_file = path
        output_file = "temp.mp4"

        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0", input_file
            ],
            stdout=subprocess.PIPE, text=True
        )
        w, h = map(int, result.stdout.strip().split(","))

        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", input_file],
            stdout=subprocess.PIPE, text=True
        )
        original_duration = float(result.stdout.strip())

        target_duration = 3600
        speed_factor = original_duration / target_duration

        scale = 400 / min(w, h)
        w = int(w * scale) // 2 * 2
        h = int(h * scale) // 2 * 2

        filters = f"setpts={1/speed_factor}*PTS,scale={w}:{h}"

        command = [
            "ffmpeg",
            "-loglevel", "warning",
            "-nostats",
            "-i", input_file,
            "-an",
            "-filter:v", filters,
            output_file
        ]

        subprocess.run(command)

        return output_file