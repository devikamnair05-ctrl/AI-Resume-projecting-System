"""
embeddings.py
Module for generating dense Transformer embeddings using sentence-transformers (all-MiniLM-L6-v2).
"""

from typing import List, Union, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Default model name
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Global cache for the loaded model instance
_CACHED_MODEL: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Retrieve or initialize the cached SentenceTransformer model instance.
    
    Args:
        model_name: HuggingFace model identifier.
        
    Returns:
        SentenceTransformer instance.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None or getattr(_CACHED_MODEL, "_model_name_or_path", "") != model_name:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CACHED_MODEL = SentenceTransformer(model_name, device=device)
    return _CACHED_MODEL


def generate_embeddings(
    texts: Union[str, List[str]],
    model: Optional[SentenceTransformer] = None,
    normalize: bool = True,
    batch_size: int = 16
) -> np.ndarray:
    """
    Compute dense vector embeddings for input text(s).
    
    Args:
        texts: A single text string or list of text strings.
        model: Optional pre-loaded SentenceTransformer instance. If None, uses cached model.
        normalize: Whether to return L2-normalized embeddings (ideal for cosine similarity).
        batch_size: Number of texts per batch during inference.
        
    Returns:
        np.ndarray of shape (N, embedding_dim) or (1, embedding_dim).
    """
    if isinstance(texts, str):
        texts = [texts]
        
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
        
    if model is None:
        model = get_embedding_model()
        
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
        convert_to_numpy=True
    )
    
    return np.array(embeddings, dtype=np.float32)
