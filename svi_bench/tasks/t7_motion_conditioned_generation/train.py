import torch, os, json
import glob
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, ImageCropAndResize, ToAbsolutePath, LoadBBox, LoadBBoxFromPlayerSpecs, PolishedCaptionsLookup, BBoxFolderDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        use_overlay_method=False,
        bbox_color_mode="noise",  # "noise" or "color"
        bbox_first_last_only=False,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.use_overlay_method = use_overlay_method
        self.bbox_color_mode = bbox_color_mode
        self.bbox_first_last_only = bbox_first_last_only
        self.optical_flow_channels = getattr(self, 'optical_flow_channels', 8)  # Default to 8
        self.background_video_channels = getattr(self, 'background_video_channels', 8)  # Default to 8
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Add bbox data if available
        if "bbox" in data:
            inputs_shared["bbox"] = data["bbox"]
            inputs_shared["use_overlay_method"] = self.use_overlay_method
            inputs_shared["bbox_color_mode"] = self.bbox_color_mode
            inputs_shared["bbox_first_last_only"] = self.bbox_first_last_only
            # Pass original video dimensions for center-crop bbox alignment
            if "orig_video_width" in data:
                inputs_shared["orig_video_width"] = data["orig_video_width"]
                inputs_shared["orig_video_height"] = data["orig_video_height"]
        
        # Add channel configurations
        inputs_shared["optical_flow_channels"] = self.optical_flow_channels
        inputs_shared["background_video_channels"] = self.background_video_channels
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    
    # Check if we should use folder-based dataset (when bbox_folder or optical_flow_folder is specified)
    if hasattr(args, 'bbox_folder') and args.bbox_folder is not None:
        # Validate required parameters for folder-based dataset
        if not hasattr(args, 'video_base_path') or args.video_base_path is None:
            raise ValueError("video_base_path is required when using bbox_folder")
        # Build captions lookup if provided
        polished_captions_lookup = None
        if getattr(args, 'captions', None):
            polished_captions_lookup = PolishedCaptionsLookup(args.captions)

        # Build LoadBBoxFromPlayerSpecs if bbox_first_last_only
        load_bbox_from_player_specs = None
        if getattr(args, 'bbox_first_last_only', False):
            load_bbox_from_player_specs = LoadBBoxFromPlayerSpecs(args.num_frames)

        # Use folder-based dataset
        dataset = BBoxFolderDataset(
            bbox_folder_path=args.bbox_folder,
            video_base_path=args.video_base_path,
            optical_flow_folder=getattr(args, 'optical_flow_folder', None),
            background_video_folder=getattr(args, 'background_video_folder', None),
            video_extension=getattr(args, 'video_extension', '.mp4'),
            prompt=getattr(args, 'prompt', 'a realistic basketball game video'),
            num_frames=args.num_frames,
            time_division_factor=getattr(args, "time_division_factor", 1),
            time_division_remainder=getattr(args, "time_division_remainder", 1),
            repeat=args.dataset_repeat,
            polished_captions_lookup=polished_captions_lookup,
            bbox_first_last_only=getattr(args, 'bbox_first_last_only', False),
        )
        
        # Create a wrapper to make it compatible with the training pipeline
        class SharedSampler:
            """Shared sampler to ensure video and bbox use the same random start index"""
            def __init__(self):
                self.last_start_idx = None
                self.orig_video_width = None
                self.orig_video_height = None
                
        class DatasetWrapper:
            def __init__(self, folder_dataset, main_data_operator, special_operator_map, load_bbox_from_player_specs=None):
                self.folder_dataset = folder_dataset
                self.main_data_operator = main_data_operator
                self.special_operator_map = special_operator_map
                self.load_from_cache = False
                self.shared_sampler = SharedSampler()
                self.load_bbox_from_player_specs = load_bbox_from_player_specs
                
                # Inject shared sampler into operators
                if hasattr(self.main_data_operator, 'shared_sampler'):
                    self.main_data_operator.shared_sampler = self.shared_sampler
                if "bbox" in self.special_operator_map:
                    self.special_operator_map["bbox"].shared_sampler = self.shared_sampler
                
            def __len__(self):
                return len(self.folder_dataset)
                
            def __getitem__(self, idx):
                # Reset shared sampler for this sample
                self.shared_sampler.last_start_idx = None
                
                data = self.folder_dataset[idx]
                
                # Process video and bbox data with shared start_idx
                # LoadVideo will generate start_idx and store in shared_sampler
                # LoadBBox will then use the same start_idx
                data["video"] = self.main_data_operator(data["video"])
                # Store original video dimensions (set by LoadVideo before ImageCropAndResize)
                if self.shared_sampler.orig_video_width is not None:
                    data["orig_video_width"] = self.shared_sampler.orig_video_width
                    data["orig_video_height"] = self.shared_sampler.orig_video_height
                if "bbox" in self.special_operator_map and "bbox" in data:
                    # If player_specifications available, construct sparse bbox from specs
                    if "player_specifications" in data and self.load_bbox_from_player_specs is not None:
                        import torch as _torch
                        bbox_data = self.load_bbox_from_player_specs(data.pop("player_specifications"), num_frames=len(data["video"]))
                        data["bbox"] = _torch.from_numpy(bbox_data)
                    else:
                        data["bbox"] = self.special_operator_map["bbox"](data["bbox"])
                elif "bbox" in data:
                    # Drop bbox if not requested
                    del data["bbox"]
                
                # Process background_video if available (similar to optical flow)
                if "background_video" in data:
                    background_video_operator = UnifiedDataset.default_video_operator(
                        base_path=args.video_base_path,
                        max_pixels=args.max_pixels,
                        height=args.height,
                        width=args.width,
                        height_division_factor=16,
                        width_division_factor=16,
                        num_frames=args.num_frames,
                        time_division_factor=args.time_division_factor,
                        time_division_remainder=args.time_division_remainder,
                    )
                    # Inject shared sampler to sync with main video (same start frame)
                    if hasattr(background_video_operator, 'shared_sampler'):
                        background_video_operator.shared_sampler = self.shared_sampler
                    # Load background video
                    data["background_video"] = background_video_operator(data["background_video"])
                
                return data
        
        # Create the wrapper dataset
        # Only include bbox operator if 'bbox' is in extra_inputs
        use_bbox = False
        if hasattr(args, 'extra_inputs') and args.extra_inputs is not None:
            try:
                use_bbox = 'bbox' in [s.strip() for s in args.extra_inputs.split(',') if s]
            except Exception:
                use_bbox = False

        dataset = DatasetWrapper(
            folder_dataset=dataset,
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.video_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=args.time_division_factor,
                time_division_remainder=args.time_division_remainder,
            ),
            special_operator_map={
                **({"bbox": LoadBBox(args.num_frames, time_division_factor=args.time_division_factor, time_division_remainder=args.time_division_remainder)} if use_bbox else {})
            },
            load_bbox_from_player_specs=load_bbox_from_player_specs,
        )
    elif hasattr(args, 'video_base_path') and args.video_base_path is not None:
        # Video-only folder dataset (no bbox). Discover videos by extension under base path.
        class VideoFolderDataset:
            def __init__(self, video_base_path, video_extension, prompt, repeat=1):
                self.video_base_path = video_base_path
                self.video_extension = video_extension if video_extension.startswith('.') else f'.{video_extension}'
                self.prompt = prompt
                self.repeat = repeat
                pattern = os.path.join(video_base_path, '**', f'*{self.video_extension}')
                self.video_paths = sorted(glob.glob(pattern, recursive=True))
                if not self.video_paths:
                    raise ValueError(f"No videos found under {video_base_path} with extension {self.video_extension}")
                self._effective_len = len(self.video_paths) * max(1, int(self.repeat))
            def __len__(self):
                return self._effective_len
            def __getitem__(self, idx):
                real_idx = idx % len(self.video_paths)
                return {
                    "video": self.video_paths[real_idx],
                    "prompt": self.prompt,
                }

        class VideoOnlyWrapper:
            def __init__(self, folder_dataset, video_operator):
                self.folder_dataset = folder_dataset
                self.video_operator = video_operator
                self.load_from_cache = False
            def __len__(self):
                return len(self.folder_dataset)
            def __getitem__(self, idx):
                data = self.folder_dataset[idx]
                data["video"] = self.video_operator(data["video"])
                return data

        folder_dataset = VideoFolderDataset(
            video_base_path=args.video_base_path,
            video_extension=getattr(args, 'video_extension', '.mp4'),
            prompt=getattr(args, 'prompt', 'a realistic basketball game video'),
            repeat=args.dataset_repeat,
        )
        dataset = VideoOnlyWrapper(
            folder_dataset=folder_dataset,
            video_operator=UnifiedDataset.default_video_operator(
                base_path=args.video_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=args.time_division_factor,
                time_division_remainder=args.time_division_remainder,
            ),
        )
    else:
        # Use original CSV-based dataset
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=args.time_division_factor,
                time_division_remainder=args.time_division_remainder,
            ),
            special_operator_map={
                "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, time_division_factor=args.time_division_factor, time_division_remainder=args.time_division_remainder, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
                "bbox": ToAbsolutePath(args.dataset_base_path) >> LoadBBox(args.num_frames, time_division_factor=args.time_division_factor, time_division_remainder=args.time_division_remainder)
            }
        )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        use_overlay_method=getattr(args, 'use_overlay_method', False),
        bbox_color_mode=getattr(args, 'bbox_color_mode', 'noise'),
        bbox_first_last_only=getattr(args, 'bbox_first_last_only', False),
    )
    
    # Set channel configurations
    model.bbox_channels = getattr(args, 'bbox_channels', 16)  # Basketball: 16, Soccer: 8
    model.optical_flow_channels = getattr(args, 'optical_flow_channels', 8)  # Default: 8
    model.background_video_channels = getattr(args, 'background_video_channels', 8)  # Default: 8
    
    # Get validation script from args or environment variable
    validation_script = getattr(args, 'validation_script', None) or os.environ.get('VALIDATION_SCRIPT', None)
    
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        validation_script=validation_script
    )
    launch_training_task(dataset, model, model_logger, args=args)
