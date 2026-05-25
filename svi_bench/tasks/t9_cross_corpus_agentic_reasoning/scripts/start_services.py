#!/usr/bin/env python
"""
Helper script to parse configs and start GPU services.
Reads from both archs/*.yaml (GPU allocation) and configs/models.yaml (model details).

Supports:
- Single instance per model (default)
- Tensor parallelism (multiple GPUs for one instance)
- Replicas (multiple instances on different ports/GPUs)

Usage:
    python scripts/start_services.py --arch gpt5 [--dry-run]
"""

import os
import sys
import yaml
import argparse
import subprocess
import signal
from typing import Dict, List, Optional, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


T9_ROOT_NOT_SET = "T9_ROOT_NOT_SET"


def _resolve_t9_data_root() -> str:
    """T9 data root. Same resolution as run_agent.py / run_batch.py."""
    if v := os.environ.get("T9_ROOT"):
        return v
    parent = os.path.dirname(os.path.abspath(__file__))
    while parent != "/" and not os.path.isfile(os.path.join(parent, "pyproject.toml")):
        parent = os.path.dirname(parent)
    for name in ("T9", "t9"):
        candidate = os.path.join(parent, "data", name)
        if os.path.isdir(candidate):
            return candidate
    return T9_ROOT_NOT_SET


def _require_t9_data_root() -> str:
    root = _resolve_t9_data_root()
    if root == T9_ROOT_NOT_SET:
        raise FileNotFoundError(
            "T9 data root not found. Either set the T9_ROOT env var or run "
            "`svi-bench download --tasks t9` to populate <repo>/data/t9/."
        )
    return root


T9_DATA_ROOT = _resolve_t9_data_root()


def _maybe_resolve_local_path(value: str) -> str:
    """Resolve a YAML model-path that's a T9-local relative path; pass through HF ids.

    Heuristic: if ``value`` starts with ``ckpts/`` or ``internvideo2/``, treat
    as a path relative to T9_ROOT or the task module dir respectively. HF
    model ids (e.g., ``Qwen/Qwen3-VL-...``) pass through unchanged.
    """
    if not value or os.path.isabs(value):
        return value
    if value.startswith("ckpts/"):
        return os.path.join(_require_t9_data_root(), value)
    if value.startswith("internvideo2/"):
        return os.path.join(BASE_DIR, value)
    return value  # HF id or other identifier


# Support experiment-specific log directory via environment variable. Logs land
# under T9_ROOT/logs by default; if T9_ROOT isn't set yet, fall back to
# BASE_DIR/logs so this module imports without raising — the actual service
# spawn will hit the loud check via _require_t9_data_root().
LOGS_DIR = os.environ.get(
    'LOGS_DIR',
    os.path.join(T9_DATA_ROOT if T9_DATA_ROOT != T9_ROOT_NOT_SET else BASE_DIR, 'logs')
)

# Suffix for log filenames so concurrent services on different nodes don't collide.
# Prefers SLURM_JOB_ID; falls back to hostname if not running under SLURM.
def _log_suffix() -> str:
    jid = os.environ.get('SLURM_JOB_ID')
    if jid:
        return f"_{jid}"
    import socket
    return f"_{socket.gethostname().split('.')[0]}"
LOG_SUFFIX = _log_suffix()


def load_yaml(path: str) -> dict:
    """Load YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def extract_port(server_url: str, default: int = 8099) -> int:
    """Extract port from server URL."""
    try:
        # http://localhost:8099/v1 -> 8099
        return int(server_url.split(':')[-1].split('/')[0])
    except (ValueError, AttributeError, IndexError):
        return default


def get_service_config(arch_name: str) -> Dict[str, List[Dict]]:
    """
    Load merged service config from arch and models.yaml.
    
    Returns dict with structure:
    {
        'llava_next_video': [
            {  # instance 0
                'enabled': True,
                'gpus': [1],
                'model': 'path/to/model',
                'port': 8099,
                'type': 'tool_model',
                'tensor_parallel_size': 1,
            },
            {  # instance 1 (replica)
                'enabled': True,
                'gpus': [2],
                'model': 'path/to/model',
                'port': 8100,
                ...
            },
        ],
        ...
    }
    """
    arch_path = os.path.join(BASE_DIR, f"archs/{arch_name}.yaml")
    models_path = os.path.join(BASE_DIR, "configs/models.yaml")
    
    if not os.path.exists(arch_path):
        raise FileNotFoundError(f"Architecture not found: {arch_path}")
    
    arch_cfg = load_yaml(arch_path)
    models_cfg = load_yaml(models_path)
    
    services = arch_cfg.get('services', {})
    merged = {}
    
    # Helper to create instance config
    def create_vllm_instance(key: str, model_info: Dict, service_cfg: Dict, 
                              instance_gpus: List[int], port: int, 
                              model_type: str = 'tool_model') -> Dict:
        
        # Merge generate_cfg: model_info (base) -> service_cfg (override)
        base_gen_cfg = model_info.get('generate_cfg', {})
        override_gen_cfg = service_cfg.get('generate_cfg', {})
        merged_gen_cfg = base_gen_cfg.copy()
        merged_gen_cfg.update(override_gen_cfg)
        
        return {
            'enabled': True,
            'gpus': instance_gpus,
            'model': _maybe_resolve_local_path(model_info.get('name')),
            'served_name': model_info.get('name'),  # unresolved name; clients use this
            'server': model_info.get('server', f'http://localhost:{port}/v1'),
            'port': port,
            'type': model_type,
            'tensor_parallel_size': service_cfg.get('tensor_parallel_size', len(instance_gpus)),
            'generate_cfg': merged_gen_cfg,
        }
    
    # Process agent model
    agent_service = services.get('agent', {})
    if agent_service.get('enabled', False):
        agent_key = arch_cfg.get('agent', {}).get('model_key')
        if agent_key and agent_key in models_cfg.get('agent_models', {}):
            model_info = models_cfg['agent_models'][agent_key]
            port = extract_port(model_info.get('server', ''), 8090)
            
            instances = process_replicas(agent_service, model_info, port, 'agent_model')
            if instances:
                merged['agent'] = instances
    
    # Process tool models and embedding models
    tool_models = models_cfg.get('tool_models', {})
    embedding_models = models_cfg.get('embedding_models', {})
    
    for key, service_cfg in services.items():
        if key == 'agent':
            continue
        if not service_cfg.get('enabled', False):
            continue
            
        # Check if it's a tool model
        if key in tool_models:
            model_info = tool_models[key]
            base_port = extract_port(model_info.get('server', ''), 8099)
            instances = process_replicas(service_cfg, model_info, base_port, 'tool_model')
            if instances:
                merged[key] = instances
                
        # Check if it's an embedding model
        elif key in embedding_models:
            model_info = embedding_models[key]
            base_port = service_cfg.get('port', 8091)
            
            # Embedding models have simpler config
            instances = []
            gpus = service_cfg.get('gpus', [0])
            replicas = service_cfg.get('replicas', None)
            
            if replicas:
                # Multiple replicas specified
                for i, replica in enumerate(replicas):
                    replica_port = replica.get('port', base_port + i)
                    replica_gpus = replica.get('gpus', [gpus[i] if i < len(gpus) else gpus[0]])
                    instances.append({
                        'enabled': True,
                        'gpus': replica_gpus if isinstance(replica_gpus, list) else [replica_gpus],
                        'config_path': _maybe_resolve_local_path(model_info.get('config_path')),
                        'model_path': _maybe_resolve_local_path(model_info.get('model_path')),
                        'port': replica_port,
                        'type': 'embedding_model',
                    })
            else:
                # Single instance
                instances.append({
                    'enabled': True,
                    'gpus': gpus if isinstance(gpus, list) else [gpus],
                    'config_path': _maybe_resolve_local_path(model_info.get('config_path')),
                    'model_path': _maybe_resolve_local_path(model_info.get('model_path')),
                    'port': base_port,
                    'type': 'embedding_model',
                })
            
            merged[key] = instances
    
    return merged


def process_replicas(service_cfg: Dict, model_info: Dict, base_port: int, 
                     model_type: str) -> List[Dict]:
    """
    Process replica configuration.
    
    Supports two formats:
    1. Simple (single instance or tensor parallel):
       gpus: [1, 2]
       tensor_parallel_size: 2
       
    2. Replicas (multiple instances):
       replicas:
         - gpus: [1]
           port: 8099
         - gpus: [2]
           port: 8100
    """
    instances = []
    gpus = service_cfg.get('gpus', [0])
    replicas = service_cfg.get('replicas', None)
    
    base_gen_cfg = model_info.get('generate_cfg', {})
    base_vllm_cfg = model_info.get('vllm_cfg', {})
    
    if replicas:
        # Multiple replicas explicitly defined
        for i, replica in enumerate(replicas):
            replica_port = replica.get('port', base_port + i)
            replica_gpus = replica.get('gpus', [gpus[i] if i < len(gpus) else gpus[0]])
            tp_size = replica.get('tensor_parallel_size', len(replica_gpus) if isinstance(replica_gpus, list) else 1)
            
            # Merge generate_cfg for replica
            metrics_cfg = base_gen_cfg.copy()
            metrics_cfg.update(service_cfg.get('generate_cfg', {})) # Arch level override
            metrics_cfg.update(replica.get('generate_cfg', {}))     # Replica level override

            # Merge vllm_cfg for replica
            vllm_cfg = base_vllm_cfg.copy()
            vllm_cfg.update(service_cfg.get('vllm_cfg', {}))
            vllm_cfg.update(replica.get('vllm_cfg', {}))
            
            instances.append({
                'enabled': True,
                'gpus': replica_gpus if isinstance(replica_gpus, list) else [replica_gpus],
                'model': _maybe_resolve_local_path(model_info.get('name')),
                'served_name': model_info.get('name'),  # unresolved name; clients use this
                'server': model_info.get('server', f'http://localhost:{replica_port}/v1'),
                'port': replica_port,
                'type': model_type,
                'tensor_parallel_size': tp_size,
                'generate_cfg': metrics_cfg,
                'vllm_cfg': vllm_cfg,
            })
    else:
        # Single instance (possibly with tensor parallelism)
        tp_size = service_cfg.get('tensor_parallel_size', 1)
        
        # Merge generate_cfg
        metrics_cfg = base_gen_cfg.copy()
        metrics_cfg.update(service_cfg.get('generate_cfg', {}))

        # Merge vllm_cfg
        vllm_cfg = base_vllm_cfg.copy()
        vllm_cfg.update(service_cfg.get('vllm_cfg', {}))
        
        pp_size = service_cfg.get('pipeline_parallel_size', 1)
        instances.append({
            'enabled': True,
            'gpus': gpus if isinstance(gpus, list) else [gpus],
            'model': _maybe_resolve_local_path(model_info.get('name')),
            'served_name': model_info.get('name'),  # unresolved name; clients use this
            'server': model_info.get('server', f'http://localhost:{base_port}/v1'),
            'port': base_port,
            'type': model_type,
            'tensor_parallel_size': tp_size,
            'pipeline_parallel_size': pp_size,
            'generate_cfg': metrics_cfg,
            'vllm_cfg': vllm_cfg,
        })
    
    return instances


def start_vllm_server(name: str, config: Dict, instance_id: int = 0,
                      dry_run: bool = False,
                      disable_flash_attn: bool = False) -> Optional[subprocess.Popen]:
    """Start a vLLM server for a model instance."""
    gpus = ','.join(map(str, config['gpus']))
    port = config['port']
    model = config['model']
    tp_size = config.get('tensor_parallel_size', 1)
    
    # Extract config for server
    vllm_cfg = config.get('vllm_cfg', {})
    gen_cfg = config.get('generate_cfg', {})

    # Priority: vllm_cfg -> generate_cfg (backwards compatibility) -> direct config
    max_len = vllm_cfg.get('max_model_len') or gen_cfg.get('max_model_len') or config.get('max_model_len')
    gpu_util = vllm_cfg.get('gpu_memory_utilization') or gen_cfg.get('gpu_memory_utilization') or '0.9'

    
    instance_name = f"{name}_{instance_id}" if instance_id > 0 else name
    
    cmd = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        '--model', model,
        '--port', str(port),
        '--host', '0.0.0.0',
        '--gpu-memory-utilization', str(gpu_util),
        '--trust-remote-code',
    ]

    # Advertise the model under its original (unresolved) name so OpenAI clients
    # that read models.yaml's `name:` field can address it directly. Without
    # this, vLLM serves under the resolved absolute path while clients send the
    # relative ckpts/... form → 404.
    served_name = config.get('served_name')
    if served_name and served_name != model:
        cmd.extend(['--served-model-name', served_name])

    if tp_size > 1:
        cmd.extend(['--tensor-parallel-size', str(tp_size)])

    pp_size = config.get('pipeline_parallel_size') or vllm_cfg.get('pipeline_parallel_size')
    if pp_size and int(pp_size) > 1:
        cmd.extend(['--pipeline-parallel-size', str(pp_size)])

    if max_len:
        cmd.extend(['--max-model-len', str(max_len)])
        
        # Check for dynamic RoPE scaling (YaRN)
        rope_scale = vllm_cfg.get('rope_scale_factor') or gen_cfg.get('rope_scale_factor')
        if rope_scale and int(max_len) > 32768:
            import json
            rope_config = {
                'rope_parameters': {
                    "rope_type": "yarn",
                    "factor": float(rope_scale),
                    "original_max_position_embeddings": 32768
                }
            }
            # Use json.dumps to ensure valid JSON string, but single quotes for shell might be tricky if handled by subprocess list
            # subprocess list arguments don't need shell escaping usually
            cmd.extend(['--hf-overrides', json.dumps(rope_config)])

    enable_prefix_caching = vllm_cfg.get('enable_prefix_caching') or gen_cfg.get('enable_prefix_caching')
    if enable_prefix_caching:
        cmd.append('--enable-prefix-caching')

    if vllm_cfg.get('enable_expert_parallel'):
        cmd.append('--enable-expert-parallel')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus
    # Redirect vLLM/flashinfer/triton cache dirs to avoid filling up /home
    cache_root = os.path.join(BASE_DIR, '.cache')
    env.setdefault('VLLM_CONFIG_ROOT', os.path.join(cache_root, 'vllm'))
    env.setdefault('FLASHINFER_WORKSPACE_BASE', BASE_DIR)
    env.setdefault('TRITON_HOME', BASE_DIR)
    if disable_flash_attn:
        env['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

    print(f"[GPU {gpus}] Starting {instance_name}: {model} on port {port} (TP={tp_size})")
    if dry_run:
        # For dry run print, we want to see the command exactly as it would be impactful
        cmd_str = ' '.join(cmd)
        # We might want to look at how to represent the json argument nicely
        print(f"  CMD: CUDA_VISIBLE_DEVICES={gpus} {cmd_str}")
        return None
    
    log_file = open(os.path.join(LOGS_DIR, f"vllm_{instance_name}{LOG_SUFFIX}.log"), 'w')
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"  PID: {proc.pid}")
    return proc


def start_embedding_server(name: str, config: Dict, instance_id: int = 0,
                           dry_run: bool = False,
                           disable_flash_attn: bool = False) -> Optional[subprocess.Popen]:
    """Start an embedding server (e.g., InternVideo2)."""
    gpus = ','.join(map(str, config['gpus']))
    port = config.get('port', 8091)
    
    instance_name = f"{name}_{instance_id}" if instance_id > 0 else name
    
    cmd = [
        'python', 'tools/serve_internvideo2.py',
        '--port', str(port),
        '--host', '0.0.0.0',
    ]
    
    if config.get('config_path'):
        cmd.extend(['--config', config['config_path']])
    if config.get('model_path'):
        cmd.extend(['--model', config['model_path']])
    if disable_flash_attn:
        cmd.append('--no-flash-attn')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus
    
    print(f"[GPU {gpus}] Starting {instance_name} embedding server on port {port}")
    if dry_run:
        print(f"  CMD: CUDA_VISIBLE_DEVICES={gpus} {' '.join(cmd)}")
        return None
    
    log_file = open(os.path.join(LOGS_DIR, f"{instance_name}{LOG_SUFFIX}.log"), 'w')
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, cwd=BASE_DIR)
    print(f"  PID: {proc.pid}")
    return proc


def main():
    parser = argparse.ArgumentParser(description="Start GPU services for batch processing")
    parser.add_argument("--arch", required=True, help="Architecture name (e.g., gpt5)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable FlashAttention for all services")
    args = parser.parse_args()
    
    print("=" * 50)
    print("VDR GPU Services")
    print("=" * 50)
    print(f"Architecture: {args.arch}")
    print(f"Base dir: {BASE_DIR}")
    print()
    
    # Create logs directory
    os.makedirs(LOGS_DIR, exist_ok=True)
    print(f"Logs dir: {LOGS_DIR}")
    
    # Get merged config
    services = get_service_config(args.arch)

    if not services:
        print("No services enabled!")
        return

    # Apply port offset for multi-experiment isolation
    port_offset = int(os.environ.get('PORT_OFFSET', '0'))
    if port_offset:
        print(f"Applying PORT_OFFSET={port_offset} to all service ports")
        for name, instances in services.items():
            for inst in instances:
                inst['port'] = inst['port'] + port_offset

    # Print summary
    print("Enabled services:")
    total_instances = 0
    for name, instances in services.items():
        for i, inst in enumerate(instances):
            prefix = f"  - {name}" if i == 0 else f"    replica {i}"
            gpus = inst['gpus']
            port = inst['port']
            tp = inst.get('tensor_parallel_size', 1)
            print(f"{prefix}: GPU {gpus}, port {port}, TP={tp}")
            total_instances += 1
    print(f"\nTotal: {total_instances} instances")
    print()
    
    # Start services
    processes = []
    
    for name, instances in services.items():
        for i, cfg in enumerate(instances):
            if cfg['type'] in ('agent_model', 'tool_model'):
                proc = start_vllm_server(name, cfg, i, args.dry_run, args.no_flash_attn)
            elif cfg['type'] == 'embedding_model':
                proc = start_embedding_server(name, cfg, i, args.dry_run, args.no_flash_attn)
            else:
                print(f"Unknown service type: {cfg['type']}")
                continue
            
            if proc:
                processes.append((f"{name}_{i}" if i > 0 else name, proc))
    
    if args.dry_run:
        print("\n[DRY RUN] No processes started")
        return
    
    print(f"\nStarted {len(processes)} service instances")
    print("Press Ctrl+C to stop all services\n")
    
    # Save PIDs and ports for load balancer or client use
    service_info = []
    for name, instances in services.items():
        ports = [inst['port'] for inst in instances]
        service_info.append(f"{name}:{','.join(map(str, ports))}")
    
    with open(os.path.join(LOGS_DIR, "service_info.txt"), 'w') as f:
        f.write('\n'.join(service_info))
    
    with open(os.path.join(LOGS_DIR, "service_pids.txt"), 'w') as f:
        for name, proc in processes:
            f.write(f"{name}:{proc.pid}\n")
    
    # Wait for processes
    def signal_handler(sig, frame):
        print("\nShutting down services...")
        for name, proc in processes:
            print(f"  Stopping {name} (PID {proc.pid})")
            proc.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Keep running
    try:
        for name, proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
