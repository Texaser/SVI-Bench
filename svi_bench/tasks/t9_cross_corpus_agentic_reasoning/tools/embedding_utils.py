from typing import Dict, Any
from llama_index.core.embeddings import BaseEmbedding

# Embedding Utilities
# Handles loading of various embedding models (InternVideo2, BGE-M3, Mock)
# avoiding circular imports and centralizing model setup.

import os
import sys
import warnings
import logging

from typing import Dict, Any, List
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
import torch

# Suppress InternVideo2 warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*flash_attn.*')
warnings.filterwarnings('ignore', message='.*deepspeed.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='timm')

# Suppress logging from InternVideo2 modules
logging.getLogger('timm').setLevel(logging.ERROR)

# InternVideo2 setup. The vendored fork lives at ../internvideo2 (one level
# up from this tools/ dir). Adding it to sys.path lets `setup_internvideo2`
# import from the upstream package layout (`configs`, `models`, `dataset`, ...).
_MM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'internvideo2'))
if _MM_PATH not in sys.path:
    sys.path.append(_MM_PATH)


try:
    import easydict
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([easydict.EasyDict])
except ImportError:
    try:
        from utils import easydict
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([easydict.EasyDict])
    except ImportError:
        print("Warning: Failed to import easydict. Checkpoint loading might fail on PyTorch 2.6+.")

def _load_module_from_path(module_name, file_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Suppress InternVideo2 internal loggers before importing
logging.getLogger('models.backbones.internvideo2.internvideo2').setLevel(logging.ERROR)

try:
    # Load Config and eval_dict_leaf from demo_config.py
    _demo_cfg_mod = _load_module_from_path("demo_config", os.path.join(_MM_PATH, "demo_config.py"))
    Config = _demo_cfg_mod.Config
    eval_dict_leaf = _demo_cfg_mod.eval_dict_leaf
    
    # Load setup_internvideo2 from demo/utils.py
    _demo_utils_mod = _load_module_from_path("demo.utils", os.path.join(_MM_PATH, "demo/utils.py"))
    setup_internvideo2 = _demo_utils_mod.setup_internvideo2

except Exception as e:
    print(f"Warning: Failed to import InternVideo2 tools: {e}. Custom embedding will fail.")
    Config = None
    eval_dict_leaf = None
    setup_internvideo2 = None

class InternVideo2Embedding(BaseEmbedding):
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _config: Any = PrivateAttr()

    def __init__(
        self,
        config_path: str = None,
        model_path: str = None,
        device: str = None,
        disable_flash_attn: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        
        # Load paths from models.yaml if not provided
        # NOTE: We hardcode the path search here as in video_tools.py, or better, expect them to be passed/handled by caller?
        # The original code had a hardcoded path fallback loop. Let's keep it for compatibility but maybe standardizing on passing args is better.
        # But `VideoEmbedding()` call in factory might not pass them if not in model_config.
        
        # Copied logic:
        # tools/embedding_utils.py -> ../configs/models.yaml. base_dir is the
        # task module dir (svi_bench/tasks/t9_.../).
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if config_path is None or model_path is None:
            try:
                import yaml
                config_file = os.path.join(base_dir, "configs/models.yaml")
                with open(config_file, 'r') as f:
                    models_cfg = yaml.safe_load(f)
                    iv2_cfg = models_cfg.get('embedding_models', {}).get('internvideo2', {})
                    if config_path is None:
                        config_path = iv2_cfg.get('config_path')
                    if model_path is None:
                        model_path = iv2_cfg.get('model_path')
            except Exception as e:
                print(f"Warning: Failed to load paths from models.yaml: {e}")

        # Resolve relative paths:
        #   - config_path: task-module-relative (vendored fork lives next to us)
        #   - model_path: T9_ROOT-relative (downloaded ckpt under data/T9/ckpts/)
        if config_path and not os.path.isabs(config_path):
            config_path = os.path.join(base_dir, config_path)
        if model_path and not os.path.isabs(model_path):
            # Use the shared T9 root resolver (raises a clear error if not found).
            # base_dir is the task module dir; _t9_root lives at its top level.
            if base_dir not in sys.path:
                sys.path.insert(0, base_dir)
            from _t9_root import require_t9_data_root
            try:
                t9_root = require_t9_data_root()
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"T9 data root not found for resolving InternVideo2 "
                    f"model_path={model_path!r}. {e}"
                ) from e
            model_path = os.path.join(t9_root, model_path)

        # Strict Check
        if config_path is None:
             raise ValueError("InternVideo2 config_path must be provided in config (models.yaml or runtime)")
        if model_path is None:
             raise ValueError("InternVideo2 model_path must be provided in config (models.yaml or runtime)")
        
        print(f"Loading InternVideo2Embedding model from config: {config_path}")
        print(f"Using weights from: {model_path}")
        
        if Config is None or setup_internvideo2 is None:
             raise ImportError("InternVideo2 tools (Config, setup_internvideo2) not available. Check dependencies and paths.")

        config = Config.from_file(config_path)
        config = eval_dict_leaf(config)
        
        # Override pretrained path in config
        config.model.vision_encoder.pretrained = model_path
        config['pretrained_path'] = model_path
        
        # Override device if provided
        if device:
            config.device = device

        # Disable FlashAttention and related fused ops
        if disable_flash_attn:
            if hasattr(config, 'model') and hasattr(config.model, 'vision_encoder'):
                config.model.vision_encoder.use_flash_attn = False
                config.model.vision_encoder.use_fused_rmsnorm = False
                config.model.vision_encoder.use_fused_mlp = False
            if hasattr(config, 'model') and hasattr(config.model, 'text_encoder'):
                config.model.text_encoder.use_flash_attn = False
            print("FlashAttention disabled via --no-flash-attn flag")

        self._config = config
        self._model, self._tokenizer = setup_internvideo2(config)
        print(f"InternVideo2Embedding model loaded on {config.device}.")

        print("========= IV2 Embedding Extraction")
        print(f"num_frames: {self._config.num_frames}")
        print(f"device: {self._config.device}")
        print("=========")


    @classmethod
    def class_name(cls) -> str:
        return "internvideo2_embedding"


    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        with torch.no_grad():
            text_input = self._tokenizer(
                text, 
                padding="max_length", 
                truncation=True, 
                max_length=self._config.max_txt_l, 
                return_tensors="pt"
            ).to(self._config.device)
            
            _, tfeat = self._model.encode_text(text_input)
            tfeat = self._model.text_proj(tfeat)
            tfeat /= tfeat.norm(dim=-1, keepdim=True)
            
            return tfeat.cpu().float().numpy()[0].tolist()


    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(t) for t in texts]

    def get_video_embedding(self, video_path: str) -> List[float]:
        """Extracts video embedding using InternVideo2."""
        import cv2
        
        try:
            # Check if utils module is loaded
            if '_demo_utils_mod' not in globals() or _demo_utils_mod is None:
                 raise ImportError("InternVideo2 utils not loaded.")
                 
            frames2tensor = _demo_utils_mod.frames2tensor

            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
            cap.release()
            
            if not frames:
                raise ValueError(f"No frames read from {video_path}")

            # Config extraction
            assert hasattr(self._config, 'num_frames'), "Config must specify 'num_frames'"
            fnum = self._config.num_frames
            device = self._config.device
            
            # Prepare Tensor
            # frames2tensor expects BGR list (cv2 default)
            vid_tensor = frames2tensor(frames, fnum=fnum, target_size=(224, 224), device=device)
            
            # Get Features
            # self._model is InternVideo2_Stage2 instance which has get_vid_feat
            with torch.no_grad():
                vid_feat = self._model.get_vid_feat(vid_tensor)
            
            return vid_feat.float().cpu().numpy()[0].tolist()

        except Exception as e:
            print(f"Error extracting video embedding for {video_path}: {e}")
            raise e

    def get_video_embeddings(self, video_paths: List[str]) -> List[List[float]]:
        """Extracts video embeddings for a batch of videos."""
        import cv2
        import torch
        
        try:
            if '_demo_utils_mod' not in globals() or _demo_utils_mod is None:
                 raise ImportError("InternVideo2 utils not loaded.")
            frames2tensor = _demo_utils_mod.frames2tensor

            # Config
            assert hasattr(self._config, 'num_frames'), "Config must specify 'num_frames'"
            fnum = self._config.num_frames
            device = self._config.device

            batch_tensors = []
            valid_indices = []
            
            # 1. Prepare Tensors
            for i, video_path in enumerate(video_paths):
                try:
                    cap = cv2.VideoCapture(video_path)
                    frames = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frames.append(frame)
                    cap.release()
                    
                    if not frames:
                        raise ValueError(f"No frames for {video_path}") 

                    # [1, T, 3, H, W]
                    t = frames2tensor(frames, fnum=fnum, target_size=(224, 224), device=device)
                    batch_tensors.append(t)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Error prepping {video_path}: {e}")
                    raise e

            if not batch_tensors:
                return []

            # 2. Batch Inference
            input_tensor = torch.cat(batch_tensors, dim=0)
            
            with torch.no_grad():
                vid_feats = self._model.get_vid_feat(input_tensor)
                
            return vid_feats.float().cpu().numpy().tolist()

        except Exception as e:
            print(f"Batch extraction failed: {e}")
            raise e


class InternVideo2RemoteEmbedding(BaseEmbedding):
    """Remote InternVideo2 embedding client that calls Flask server via HTTP."""
    
    _server_url: str = PrivateAttr()
    
    def __init__(self, server_url: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._server_url = server_url.rstrip('/')
        print(f"InternVideo2RemoteEmbedding initialized with server: {self._server_url}")
    
    @classmethod
    def class_name(cls) -> str:
        return "internvideo2_remote_embedding"
    
    def _get_text_embedding(self, text: str) -> List[float]:
        import requests
        try:
            response = requests.post(
                f"{self._server_url}/embed",
                json={"text": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            raise RuntimeError(f"InternVideo2 remote embedding failed: {e}")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        import requests
        try:
            response = requests.post(
                f"{self._server_url}/embed",
                json={"texts": texts},
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            raise RuntimeError(f"InternVideo2 remote batch embedding failed: {e}")


def get_embedding_model(model_name: str, model_config: Dict = None) -> BaseEmbedding:
    """Factory to create embedding model instances.
    
    For internvideo2, checks if a server URL is configured to decide between
    local (in-process) or remote (HTTP client) mode.
    """
    # Normalize
    if not model_name: model_name = "m3"
    model_name = model_name.lower()

    if model_name == "internvideo2":
        # Remote-mode server URL is supplied by the caller (run_agent /
        # run_batch already runs it through apply_host_rewrite). Never
        # re-read models.yaml here — that bypasses the host rewrite and
        # ends up pointing at localhost from a worker node.
        server_url = model_config.get('server') if model_config else None

        # Use remote mode if server URL is configured AND reachable
        if server_url:
            try:
                import requests
                health_url = f"{server_url.rstrip('/')}/health"
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200:
                    print(f"InternVideo2 server detected at {server_url}, using remote mode")
                    return InternVideo2RemoteEmbedding(server_url=server_url)
            except Exception:
                print(f"InternVideo2 server at {server_url} not reachable, falling back to local mode")
        
        # Local mode (load model in-process) 
        return InternVideo2Embedding()

        
    elif model_name == "m3" or model_name == "bge-m3":
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            # Distribute across GPUs specified by EMBEDDING_GPUS env var
            device = None
            embedding_gpus = os.environ.get('EMBEDDING_GPUS', '')
            if embedding_gpus:
                gpu_list = [int(g) for g in embedding_gpus.split(',')]
                worker_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
                gpu_id = gpu_list[worker_id % len(gpu_list)]
                device = f"cuda:{gpu_id}"
                print(f"bge-m3: worker {worker_id} -> {device}")
            return HuggingFaceEmbedding(model_name="BAAI/bge-m3", embed_batch_size=128, device=device)
        except Exception as e:
            print(f"Error loading BAAI/bge-m3: {e}. Falling back to Mock.")
            
    elif model_name == "mock":
        from llama_index.core.embeddings import MockEmbedding
        return MockEmbedding(embed_dim=1024)
        
    # Default/Fallback
    print(f"Warning: Unknown or failed embedding model '{model_name}'. Using MockEmbedding.")
    from llama_index.core.embeddings import MockEmbedding
    return MockEmbedding(embed_dim=1024)

