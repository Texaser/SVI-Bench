from evaluation.models.model import Model

import os
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

class Qwen(Model):
    def __init__(self, num_frames=768):
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        
        checkpoint_path = "Qwen/Qwen3-VL-32B-Instruct"
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        self.llm = LLM(
            model=checkpoint_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            enforce_eager=False,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0
        )
        self.num_frames = num_frames

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
            top_k=-1,
            stop_token_ids=[],
        )

    def name(self):
        return "Qwen3-VL-32B"

    def process_question(self, prompt, video_path):
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "nframes": self.num_frames,
                },
                {"type": "text", "text": prompt},
            ],
        }]
        
        inputs = [self.prepare_inputs_for_vllm(message) for message in [messages]]

        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        response = outputs[0].outputs[0].text

        return response

    def prepare_inputs_for_vllm(self, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # qwen_vl_utils 0.0.14+ reqired
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        
        print(f"video_kwargs: {video_kwargs}")

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs

        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }

