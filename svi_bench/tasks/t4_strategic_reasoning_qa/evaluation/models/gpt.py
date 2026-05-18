from evaluation.models.model import Model

import cv2
import numpy as np
import base64
from openai import OpenAI

class GPT(Model):

    def __init__(self, key):
        self.client = OpenAI(api_key=key)

    def name(self):
        return "GPT-5.2"
    
    def process_question(self, prompt, video_path):
        frames = self.sample_frames(video_path)
        response = self.client.responses.create(
            model="gpt-5.2",
            input = [
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": (prompt) },
                        *[
                            { "type": "input_image", "image_url": f"data:image/jpeg;base64,{frame}" }
                            for frame in frames
                        ]   
                    ]
                }
            ]
        )

        return response.output_text

    def sample_frames(self, path, num_samples=500):
        video = cv2.VideoCapture(path)

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = 400/min(height, width)
        width = int(width*scale)
        height = int(height*scale)

        num_samples = min(num_samples, total_frames)

        target_indices = np.linspace(
            0, total_frames - 1, num_samples, dtype=int
        )

        frames = []
        for idx in target_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video.read()
            if not success:
                continue
                
            frame = cv2.resize(frame, (width, height))
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()

        return frames