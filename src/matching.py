"""
matching.py
Module for calculating similarity scores between Job Descriptions and Candidate Resumes.
Supports pure Cosine Similarity as well as configurable Hybrid Scoring (Semantic + Skills + Experience).
"""

from typing import Union, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(
    jd_embedding: np.ndarray,
    resume_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between a Job Description vector and Resume vectors.
    
    Args:
        jd_embedding: 1D or 2D array of shape (embedding_dim,) or (1, embedding_dim).
        resume_embeddings: 2D array of shape (num_candidates, embedding_dim).
        
    Returns:
        1D np.ndarray of similarity scores in range [0.0, 1.0] of length num_candidates.
    """
    # Ensure 2D arrays
    if jd_embedding.ndim == 1:
        jd_embedding = jd_embedding.reshape(1, -1)
    if resume_embeddings.ndim == 1:
        resume_embeddings = resume_embeddings.reshape(1, -1)
        
    if jd_embedding.size == 0 or resume_embeddings.size == 0:
        return np.array([], dtype=np.float32)
        
    # Calculate cosine similarity matrix of shape (1, num_candidates)
    sim_matrix = cosine_similarity(jd_embedding, resume_embeddings)
    scores = sim_matrix[0]
    
    # Clip to [0.0, 1.0] in case of tiny numerical artifacts
    scores = np.clip(scores, 0.0, 1.0)
    return scores


def compute_hybrid_score(
    semantic_score: float,
    skill_match_ratio: float,
    experience_fit_ratio: float = 1.0,
    weights: Tuple[float, float, float] = (0.70, 0.20, 0.10)
) -> float:
    """
    Compute a weighted composite match score combining:
    1. Semantic similarity (dense transformer embedding match)
    2. Exact/fuzzy skill overlap ratio
    3. Experience fit ratio
    
    Args:
        semantic_score: Semantic cosine similarity (0.0 to 1.0)
        skill_match_ratio: Ratio of matched skills to total required skills (0.0 to 1.0)
        experience_fit_ratio: Candidate experience / required experience ratio (capped at 1.0)
        weights: Tuple of (semantic_w, skill_w, exp_w) summing to 1.0
        
    Returns:
        Final composite score as a percentage (0.0 to 100.0)
    """
    w_sem, w_skill, w_exp = weights
    total_w = w_sem + w_skill + w_exp
    if total_w <= 0:
        total_w = 1.0
    w_sem, w_skill, w_exp = w_sem / total_w, w_skill / total_w, w_exp / total_w
    
    composite = (
        (semantic_score * w_sem) +
        (skill_match_ratio * w_skill) +
        (experience_fit_ratio * w_exp)
    )
    
    # Return as percentage rounded to 2 decimal places
    return round(float(np.clip(composite * 100.0, 0.0, 100.0)), 2)
