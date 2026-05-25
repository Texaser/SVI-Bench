import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image



class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=1, time_division_remainder=1, frame_processor=lambda x: x, shared_sampler=None, random_sample=False):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        self.shared_sampler = shared_sampler  # Shared sampler to sync with bbox
        self.random_sample = random_sample  # Whether to use random sampling
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        import numpy as np
        reader = imageio.get_reader(data)
        total_frames = int(reader.count_frames())
        
        # Calculate sampling parameters
        step = self.time_division_factor
        max_start = total_frames - self.num_frames * step
        
        # Determine start index
        if self.random_sample and max_start > 0:
            # Random sampling: use shared sampler if available to sync with bbox
            if self.shared_sampler is not None and hasattr(self.shared_sampler, 'last_start_idx') and self.shared_sampler.last_start_idx is not None:
                start_idx = self.shared_sampler.last_start_idx
            else:
                start_idx = np.random.randint(0, max_start + 1)
                # Store for bbox to use
                if self.shared_sampler is not None:
                    self.shared_sampler.last_start_idx = start_idx
        else:
            # Default: start from beginning
            start_idx = 0
            if self.shared_sampler is not None:
                self.shared_sampler.last_start_idx = start_idx
        
        # Sample frames with stride
        frames = []
        for i in range(self.num_frames):
            frame_id = start_idx + i * step
            if frame_id < total_frames:
                frame = reader.get_data(frame_id)
                frame = Image.fromarray(frame)
                # Store original video dimensions (before crop/resize) on shared_sampler
                if i == 0 and self.shared_sampler is not None:
                    self.shared_sampler.orig_video_width = frame.size[0]
                    self.shared_sampler.orig_video_height = frame.size[1]
                frame = self.frame_processor(frame)
                frames.append(frame)
            else:
                # Video has insufficient frames, pad with the last available frame
                if frames:
                    frames.append(frames[-1])  # Repeat last frame
                else:
                    # This shouldn't happen, but handle edge case
                    print(f"Warning: Video {data} has no frames. Skipping.")
                    reader.close()
                    return None
        
        reader.close()
        
        # Ensure we have exactly num_frames
        if len(frames) != self.num_frames:
            print(f"Warning: Expected {self.num_frames} frames but got {len(frames)} from {data}")
            # Pad with last frame if needed
            while len(frames) < self.num_frames and frames:
                frames.append(frames[-1])
            # Truncate if somehow we have too many (shouldn't happen with current logic)
            frames = frames[:self.num_frames]
        
        return frames



class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadBBox:
    """Load bbox data from npz or txt files"""
    def __init__(self, num_frames, time_division_factor=1, time_division_remainder=1, frame_processor=None, shared_sampler=None, random_sample=False):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.shared_sampler = shared_sampler  # Shared sampler to sync with video
        self.random_sample = random_sample  # Whether to use random sampling
    
    def _load_bbox_from_txt(self, txt_path):
        """Load bbox data from txt file format: frame_id,object_id,x1,y1,x2,y2,confidence,..."""
        import numpy as np
        
        # Parse all lines
        bbox_dict = {}  # {(frame_id, object_id): [x1, y1, x2, y2]}
        max_frame_id = -1
        unique_object_ids = set()
        
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                try:
                    frame_id = int(parts[0])
                    object_id = int(parts[1])
                    x1 = float(parts[2])
                    y1 = float(parts[3])
                    x2 = float(parts[4])
                    y2 = float(parts[5])
                    
                    max_frame_id = max(max_frame_id, frame_id)
                    unique_object_ids.add(object_id)
                    
                    bbox_dict[(frame_id, object_id)] = [x1, y1, x2, y2]
                except (ValueError, IndexError):
                    continue
        
        if max_frame_id < 0 or len(unique_object_ids) == 0:
            raise ValueError(f"No valid bbox data found in {txt_path}")
        
        # Create mapping from object_id to array index
        # Sort object_ids to ensure consistent ordering
        sorted_object_ids = sorted(unique_object_ids)
        object_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(sorted_object_ids)}
        num_objects = len(unique_object_ids)
        
        # Create array: (num_frames, num_objects * 4)
        num_frames = max_frame_id + 1
        bbox_data = np.zeros((num_frames, num_objects * 4), dtype=np.float32)
        
        # Fill in the data using the mapping
        for (frame_id, object_id), coords in bbox_dict.items():
            if frame_id < num_frames and object_id in object_id_to_idx:
                array_idx = object_id_to_idx[object_id]
                start_idx = array_idx * 4
                bbox_data[frame_id, start_idx:start_idx+4] = coords
        
        return bbox_data
        
    def __call__(self, data):
        import numpy as np
        import torch
        
        if isinstance(data, str):
            if data.endswith('.txt'):
                bbox_data = self._load_bbox_from_txt(data)
            elif data.endswith('.npz'):
                bbox_data = np.load(data)['arr_0']
            else:
                raise ValueError(f"Unsupported bbox file format: {data}. Expected .npz or .txt")
        else:
            bbox_data = data
        
        total_frames = len(bbox_data)
        
        # Calculate sampling parameters (same as LoadVideo)
        step = self.time_division_factor
        max_start = total_frames - self.num_frames * step
        
        # Determine start index (same logic as LoadVideo)
        if self.random_sample and max_start > 0:
            # Random sampling: use shared sampler if available to sync with video
            if self.shared_sampler is not None and hasattr(self.shared_sampler, 'last_start_idx') and self.shared_sampler.last_start_idx is not None:
                start_idx = self.shared_sampler.last_start_idx
            else:
                start_idx = np.random.randint(0, max_start + 1)
                # Store for video to use (if bbox is processed first)
                if self.shared_sampler is not None:
                    self.shared_sampler.last_start_idx = start_idx
        else:
            # Default: start from beginning
            start_idx = 0
            if self.shared_sampler is not None:
                self.shared_sampler.last_start_idx = start_idx
        
        # Sample frames with stride (same as LoadVideo)
        selected_data = []
        original_bbox_dim = bbox_data.shape[1]  # Save original dimension for fallback
        
        for i in range(self.num_frames):
            frame_id = start_idx + i * step
            if frame_id < total_frames:
                selected_data.append(bbox_data[frame_id])
            else:
                # If frame_id exceeds total_frames, pad with the last available frame or zeros
                if selected_data:
                    # Use the last available frame for padding
                    selected_data.append(selected_data[-1])
                else:
                    # If no frames were selected at all, use zeros
                    selected_data.append(np.zeros(original_bbox_dim, dtype=bbox_data.dtype))
        
        # Ensure we have exactly num_frames (same logic as LoadVideo)
        if len(selected_data) != self.num_frames:
            print(f"Warning: Expected {self.num_frames} bbox frames but got {len(selected_data)} from {data}")
            # Pad with last frame if we have fewer frames
            while len(selected_data) < self.num_frames and selected_data:
                selected_data.append(selected_data[-1])
            # Truncate if we have too many frames
            selected_data = selected_data[:self.num_frames]
        
        # Convert list to array - now we should always have exactly num_frames
        if len(selected_data) > 0:
            bbox_data = np.stack(selected_data, axis=0)
        else:
            # Fallback: create zeros array with correct shape
            bbox_data = np.zeros((self.num_frames, original_bbox_dim), dtype=bbox_data.dtype)
                
        # Convert to tensor
        bbox_data = torch.from_numpy(bbox_data).float()
        
        if self.frame_processor is not None:
            bbox_data = self.frame_processor(bbox_data)
            
        return bbox_data


class PolishedCaptionsLookup:
    """Loads polished captions JSONs and provides lookup by mixsort path.

    Builds a mapping from the relative path (shared between mixsort paths and JSON keys)
    to the caption entry. For 22-23 season: relative path after "22-23/" in JSON key matches
    relative path after "basketball_mixsort_all_22_23_season/" in mixsort path (with .mp4/.txt swap).
    For 23-24 season: relative path after "clips/" matches after "basketball_mixsort_all_23_24_season/".
    """
    def __init__(self, json_paths):
        self.entries = {}  # relative_path (no ext) -> entry dict
        for json_path in json_paths:
            with open(json_path) as f:
                data = json.load(f)
            for mp4_key, entry in data.items():
                # Extract the relative path that's shared with mixsort
                if "/22-23/" in mp4_key:
                    rel = mp4_key.split("22-23/", 1)[1]
                elif "/clips/" in mp4_key:
                    rel = mp4_key.split("clips/", 1)[1]
                else:
                    continue
                # Store without extension as the canonical key
                rel_no_ext = os.path.splitext(rel)[0]
                self.entries[rel_no_ext] = entry
        print(f"[PolishedCaptionsLookup] Loaded {len(self.entries)} entries from {len(json_paths)} files")

    def get_entry(self, mixsort_path):
        """Look up a polished caption entry by mixsort txt path.

        Returns (prompt, player_specifications) or (None, None) if not found.
        """
        # Extract relative path from mixsort path
        normalized = os.path.normpath(mixsort_path)
        parts = normalized.split(os.sep)

        # Find "basketball_mixsort_all_*" directory and take everything after it
        mixsort_idx = None
        for i, part in enumerate(parts):
            if 'mixsort_all' in part:
                mixsort_idx = i
                break

        if mixsort_idx is None:
            return None, None

        relative_parts = parts[mixsort_idx + 1:]
        rel = os.path.join(*relative_parts) if relative_parts else ""
        rel_no_ext = os.path.splitext(rel)[0]

        entry = self.entries.get(rel_no_ext)
        if entry is None:
            return None, None

        # Prompt fallback: refined_instruction > instruction > gpt_caption > caption
        instruction = (entry.get("refined_instruction")
                       or entry.get("instruction")
                       or entry.get("gpt_caption")
                       or entry.get("caption")
                       or "a realistic basketball game video")

        player_specs = entry.get("player_specifications")

        # Combine instruction + raw player_specifications JSON into prompt
        if player_specs:
            prompt = "Instructions: " + instruction + " Player_specifications: " + json.dumps(player_specs)
        else:
            prompt = "Instructions: " + instruction

        return prompt, player_specs


class LoadBBoxFromPlayerSpecs:
    """Constructs a sparse bbox tensor from player_specifications.

    Only populates frame 0 (start_bbox) and frame N-1 (end_bbox) for each player.
    All other frames are zeros.

    Output shape: (num_frames, num_players * 4)
    """
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, player_specifications, num_frames=None):
        import numpy as np

        num_frames = num_frames or self.num_frames
        num_players = len(player_specifications)
        bbox_data = np.zeros((num_frames, num_players * 4), dtype=np.float32)

        for i, spec in enumerate(player_specifications):
            start = spec.get("start_bbox", {})
            end = spec.get("end_bbox", {})

            start_coords = [start.get("x1", 0), start.get("y1", 0),
                            start.get("x2", 0), start.get("y2", 0)]
            end_coords = [end.get("x1", 0), end.get("y1", 0),
                          end.get("x2", 0), end.get("y2", 0)]

            col_start = i * 4
            bbox_data[0, col_start:col_start + 4] = start_coords
            bbox_data[num_frames - 1, col_start:col_start + 4] = end_coords

        return bbox_data


class BBoxFolderDataset(torch.utils.data.Dataset):
    """Dataset that loads bbox files (npz or txt) from a folder and generates corresponding video paths
    
    Supports two modes:
    1. bbox-first (legacy): Scan bbox_folder, generate video and optional optical_flow paths
    2. optical_flow-first (new): Scan optical_flow_folder, generate video and bbox paths
    
    Bbox file formats supported:
    - .npz: numpy array format with shape (num_frames, num_objects * 4)
    - .txt: text format with lines: frame_id,object_id,x1,y1,x2,y2,confidence,...
    """
    def __init__(
        self,
        bbox_folder_path=None,
        video_base_path=None,
        optical_flow_folder=None,
        background_video_folder=None,
        video_extension=".mp4",
        prompt="a realistic basketball game video",
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        repeat=1,
        polished_captions_lookup=None,
        bbox_first_last_only=False,
    ):
        self.bbox_folder_path = bbox_folder_path
        self.video_base_path = video_base_path
        self.optical_flow_folder = optical_flow_folder
        self.background_video_folder = background_video_folder
        self.video_extension = video_extension
        self.prompt = prompt
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.repeat = repeat
        self.polished_captions_lookup = polished_captions_lookup
        self.bbox_first_last_only = bbox_first_last_only
        
        # Determine mode: optical_flow-first or bbox-first
        if optical_flow_folder is not None and os.path.exists(optical_flow_folder):
            # Mode 2: optical_flow-first (for soccer)
            self.mode = "optical_flow_first"
            self._init_optical_flow_first()
        elif bbox_folder_path is not None:
            # Mode 1: bbox-first (legacy, for basketball)
            self.mode = "bbox_first"
            self._init_bbox_first()
        else:
            raise ValueError("Either bbox_folder_path or optical_flow_folder must be specified")
        
        # Create loaders
        self.bbox_loader = LoadBBox(num_frames, time_division_factor, time_division_remainder)
    
    def _init_bbox_first(self):
        """Legacy mode: scan bbox folder, generate video and optical_flow paths
        
        Supports two input formats:
        1. Directory: scan directory for .npz and .txt files
        2. Text file: read bbox paths from txt file (one path per line)
        """
        self.bbox_files = []
        
        # Check if bbox_folder_path is a file (txt list) or directory
        if os.path.isfile(self.bbox_folder_path):
            # Read bbox paths from txt file (one path per line)
            print(f"[BBox-First Mode] Reading bbox paths from file: {self.bbox_folder_path}")
            with open(self.bbox_folder_path, 'r') as f:
                for line in f:
                    bbox_path = line.strip()
                    if bbox_path and (bbox_path.endswith('.npz') or bbox_path.endswith('.txt')):
                        if os.path.exists(bbox_path):
                            self.bbox_files.append(bbox_path)
                        else:
                            print(f"  ⚠ Bbox file not found: {bbox_path}")
            self.bbox_folder_is_file = True
        else:
            # Scan directory for bbox files
            for root, dirs, files in os.walk(self.bbox_folder_path, followlinks=True):
                for file in files:
                    if file.endswith('.npz') or file.endswith('.txt'):
                        self.bbox_files.append(os.path.join(root, file))
            self.bbox_folder_is_file = False
        
        print(f"[BBox-First Mode] Found {len(self.bbox_files)} bbox files")
    
    def _init_optical_flow_first(self):
        """New mode: scan optical_flow folder, generate video and bbox paths"""
        self.optical_flow_files = []
        for root, dirs, files in os.walk(self.optical_flow_folder, followlinks=True):
            for file in files:
                if file.endswith('.npz'):
                    self.optical_flow_files.append(os.path.join(root, file))
        
        print(f"[OpticalFlow-First Mode] Found {len(self.optical_flow_files)} optical flow files in {self.optical_flow_folder}")
        
    def __len__(self):
        if self.mode == "bbox_first":
            return len(self.bbox_files) * self.repeat
        else:
            return len(self.optical_flow_files) * self.repeat
        
    def __getitem__(self, idx):
        if self.mode == "bbox_first":
            return self._get_item_bbox_first(idx)
        else:
            return self._get_item_optical_flow_first(idx)
    
    def _get_item_bbox_first(self, idx):
        """Legacy mode: bbox → video + optical_flow"""
        # Get the actual file index (accounting for repeat)
        file_idx = idx % len(self.bbox_files)
        bbox_file = self.bbox_files[file_idx]
        
        # Generate video path by replacing bbox folder with video folder and changing extension
        if getattr(self, 'bbox_folder_is_file', False):
            # bbox_file is an absolute path read from a list. Extract a relative path
            # we can join onto video_base_path / background_video_folder.
            # We accept two layouts:
            #   1. SVI-Bench public layout: .../bboxes/{bucket}/{ID}.txt
            #      → relative = {bucket}/{ID}.txt
            #   2. Legacy mixsort layout: .../basketball_mixsort_all_*/{league}/{game}/{name}.txt
            #      → relative = {league}/{game}/{name}.txt (everything after the mixsort dir)
            bbox_path_normalized = os.path.normpath(bbox_file)
            parts = bbox_path_normalized.split(os.sep)

            marker_idx = None
            for i, part in enumerate(parts):
                if part == 'bboxes' or 'mixsort_all' in part:
                    marker_idx = i
                    break

            if marker_idx is not None:
                relative_parts = parts[marker_idx + 1:]
                relative_path = os.path.join(*relative_parts) if relative_parts else ""
            else:
                # Fallback: basename only (loses any bucket dir, but better than crashing)
                relative_path = os.path.basename(bbox_file)
        else:
            # Original logic: bbox_file is relative to bbox_folder_path
            relative_path = os.path.relpath(bbox_file, self.bbox_folder_path)
        
        # Remove .npz or .txt extension and add video extension
        video_relative_path = os.path.splitext(relative_path)[0] + self.video_extension
        video_path = os.path.join(self.video_base_path, video_relative_path)
        
        # Create data dict
        data = {
            "video": video_path,
            "prompt": self.prompt,
            "bbox": bbox_file,
        }

        # Per-video prompt and player_specifications from polished captions
        if self.polished_captions_lookup is not None:
            prompt, player_specs = self.polished_captions_lookup.get_entry(bbox_file)
            if prompt is not None:
                data["prompt"] = prompt
            if player_specs is not None and self.bbox_first_last_only:
                data["player_specifications"] = player_specs

        # Add optical_flow path if optical_flow_folder is specified
        if self.optical_flow_folder is not None:
            optical_flow_path = os.path.join(self.optical_flow_folder, relative_path)
            data["optical_flow"] = optical_flow_path

        # Add background_video path if background_video_folder is specified (replaces optical_flow)
        if self.background_video_folder is not None:
            background_video_path = os.path.join(self.background_video_folder, video_relative_path)
            data["background_video"] = background_video_path

        return data
    
    def _get_item_optical_flow_first(self, idx):
        """New mode: optical_flow → video + bbox"""
        # Get the actual file index (accounting for repeat)
        file_idx = idx % len(self.optical_flow_files)
        optical_flow_file = self.optical_flow_files[file_idx]
        
        # Get relative path from optical_flow folder
        relative_path = os.path.relpath(optical_flow_file, self.optical_flow_folder)
        
        # Generate video path
        video_relative_path = os.path.splitext(relative_path)[0] + self.video_extension
        video_path = os.path.join(self.video_base_path, video_relative_path)
        
        # Generate bbox path (same structure as video)
        bbox_path = os.path.join(self.bbox_folder_path, relative_path)
        
        # Create data dict
        data = {
            "video": video_path,
            "prompt": self.prompt,
            "bbox": bbox_path,
            "optical_flow": optical_flow_file,
        }
        
        # Add background_video path if background_video_folder is specified (replaces optical_flow)
        if self.background_video_folder is not None:
            background_video_path = os.path.join(self.background_video_folder, video_relative_path)
            data["background_video"] = background_video_path
        
        return data


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
