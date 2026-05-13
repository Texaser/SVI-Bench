from evaluation.models.model import Model

from transformers import AutoProcessor, AutoModelForImageTextToText
from molmo_utils import process_vision_info

class Molmo(Model):
    def __init__(self):
        model_path="allenai/Molmo2-8B"

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
        )

    def name(self):
        return "Molmo2-8B"
    
    def process_question(self, prompt, video_path):
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text", "text": prompt
                },
                {
                    "type": "video",
                    "video": video_path,
                    "frame_sampling_mode": "uniform_last_frame",
                    "num_frames": 300,
                }
            ],
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        images, videos, video_kwargs = process_vision_info(messages)

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos = list(videos)
            video_metadatas = list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_text = self.processor.post_process_image_text_to_text(
            generated_ids[:, inputs["input_ids"].size(1):],
            skip_special_tokens=True,
        )[0]
        
        return generated_text

