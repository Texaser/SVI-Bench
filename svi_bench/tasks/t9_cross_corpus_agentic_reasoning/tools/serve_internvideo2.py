#!/usr/bin/env python
"""
InternVideo2 embedding server for text query encoding.
Provides HTTP API for generating text embeddings at search time.

Uses the existing InternVideo2Embedding class from embedding_utils.py.

Usage:
    CUDA_VISIBLE_DEVICES=4 python serve_internvideo2.py --port 8091 --host 0.0.0.0
"""

import os
import sys
import argparse
import numpy as np
from flask import Flask, request, jsonify
from typing import List, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent path for imports
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(TOOL_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, TOOL_DIR)

app = Flask(__name__)

# Global model reference
_model = None


def load_model(config_path: str = None, model_path: str = None, device: str = None,
               disable_flash_attn: bool = False):
    """
    Load InternVideo2 model using existing InternVideo2Embedding class.
    """
    global _model

    logger.info("Loading InternVideo2 embedding model...")

    try:
        from embedding_utils import InternVideo2Embedding

        # Create the embedding model
        # It will load config/model paths from models.yaml if not provided
        _model = InternVideo2Embedding(
            config_path=config_path,
            model_path=model_path,
            device=device,
            disable_flash_attn=disable_flash_attn,
        )
        
        logger.info("InternVideo2 model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load InternVideo2 model: {e}")
        import traceback
        traceback.print_exc()
        return False


def encode_text(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Encode text(s) to embedding(s).
    
    Args:
        texts: Single text string or list of strings
        
    Returns:
        numpy array of shape (N, embedding_dim)
    """
    global _model
    
    if isinstance(texts, str):
        texts = [texts]
    
    # Mock mode for testing
    if _model == "MOCK":
        logger.warning("MOCK mode: returning random embeddings")
        return np.random.randn(len(texts), 1024).astype(np.float32)
    
    if _model is None:
        raise RuntimeError("Model not loaded")
    
    # Use the InternVideo2Embedding's _get_text_embeddings
    embeddings = _model._get_text_embeddings(texts)
    
    return np.array(embeddings, dtype=np.float32)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': _model is not None and _model != "MOCK",
        'mock_mode': _model == "MOCK"
    })


@app.route('/embed', methods=['POST'])
def embed():
    """
    Embed text query.
    
    Request body:
        {"text": "query string"} or {"texts": ["query1", "query2"]}
    
    Response:
        {"embedding": [...]} or {"embeddings": [[...], [...]]}
    """
    try:
        data = request.json
        
        if 'text' in data:
            text = data['text']
            embedding = encode_text(text)
            return jsonify({'embedding': embedding[0].tolist()})
        
        elif 'texts' in data:
            texts = data['texts']
            embeddings = encode_text(texts)
            return jsonify({'embeddings': embeddings.tolist()})
        
        else:
            return jsonify({'error': 'Missing "text" or "texts" in request'}), 400
    
    except Exception as e:
        logger.error(f"Error in /embed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/embed_batch', methods=['POST'])
def embed_batch():
    """Batch embedding endpoint (alias for /embed with texts)."""
    return embed()


def main():
    parser = argparse.ArgumentParser(description="InternVideo2 Embedding Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8091, help="Port to listen on")
    parser.add_argument("--config", default=None, help="Path to InternVideo2 config")
    parser.add_argument("--model", default=None, help="Path to InternVideo2 checkpoint")
    parser.add_argument("--device", default=None, help="Device to use (e.g., 'cuda:0')")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (random embeddings)")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable FlashAttention (and fused ops) for the model")
    args = parser.parse_args()
    
    if args.mock:
        global _model
        _model = "MOCK"
        logger.warning("Running in MOCK mode - returning random embeddings")
    else:
        success = load_model(
            config_path=args.config,
            model_path=args.model,
            device=args.device,
            disable_flash_attn=args.no_flash_attn,
        )
        if not success:
            logger.error("Failed to load model. Use --mock for testing without model.")
            return
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
