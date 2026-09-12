"""
ranking.py
Module for ranking candidates based on semantic similarity and XAI metrics.
Produces sorted leaderboards and formatted DataFrames.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.preprocessing import clean_text
from src.embeddings import generate_embeddings, get_embedding_model
from src.matching import compute_cosine_similarity, compute_hybrid_score
from src.explainability import explain_candidate_match


def rank_candidates(
    candidates: List[Dict[str, Any]],
    job_info: Dict[str, Any],
    use_hybrid: bool = False,
    weights: Tuple[float, float, float] = (0.70, 0.20, 0.10),
    model = None
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Rank a list of candidates against a target job requirement.
    
    Args:
        candidates: List of dicts, each with at least 'raw_text' and optional 'candidate_name', 'filename'.
        job_info: Dictionary of target job details.
        use_hybrid: Whether to rank by composite hybrid score (True) or pure semantic score (False).
        weights: Weights for (semantic, skill, experience) if use_hybrid is True.
        model: Pre-loaded SentenceTransformer model instance.
        
    Returns:
        Tuple of:
        1. List of ranked candidate dicts (sorted descending by score)
        2. Clean pandas DataFrame formatted for display / export.
    """
    if not candidates:
        return [], pd.DataFrame()
        
    if model is None:
        model = get_embedding_model()
        
    # Construct target job text
    jd_full_text = job_info.get("full_job_text", "")
    if not jd_full_text:
        jd_full_text = (
            f"Title: {job_info.get('job_title', '')}. "
            f"Skills: {job_info.get('required_skills', '')}. "
            f"Experience: {job_info.get('experience_years', '')}. "
            f"Description: {job_info.get('job_description', '')}"
        )
        
    clean_jd = clean_text(jd_full_text)
    jd_embedding = generate_embeddings(clean_jd, model=model)
    
    # Preprocess all candidates
    clean_resumes = []
    for c in candidates:
        text = c.get("raw_text", "")
        clean_resumes.append(clean_text(text))
        
    # Generate embeddings in batch
    resume_embeddings = generate_embeddings(clean_resumes, model=model)
    
    # Calculate cosine similarities
    cosine_scores = compute_cosine_similarity(jd_embedding, resume_embeddings)
    
    results = []
    
    for i, candidate in enumerate(candidates):
        sem_score_pct = round(float(cosine_scores[i]) * 100.0, 2)
        raw_text = candidate.get("raw_text", "")
        
        # Explainability analysis
        xai = explain_candidate_match(raw_text, job_info, sem_score_pct)
        
        # Hybrid score calculation
        hybrid_score_pct = compute_hybrid_score(
            semantic_score=float(cosine_scores[i]),
            skill_match_ratio=xai["skill_match_ratio"],
            experience_fit_ratio=xai["experience_ratio"],
            weights=weights
        )
        
        ranking_score = hybrid_score_pct if use_hybrid else sem_score_pct
        
        entry = {
            "candidate_name": candidate.get("candidate_name", f"Candidate {i+1}"),
            "filename": candidate.get("filename", "uploaded_resume"),
            "semantic_score": sem_score_pct,
            "hybrid_score": hybrid_score_pct,
            "final_score": ranking_score,
            "qualification": xai["qualification_category"],
            "qualification_badge": xai["qualification_badge"],
            "matched_skills": xai["matched_skills"],
            "missing_skills": xai["missing_skills"],
            "skill_match_pct": round(xai["skill_match_ratio"] * 100, 1),
            "experience_status": xai["experience_status"],
            "candidate_exp_years": xai["candidate_exp_years"],
            "summary": xai["summary"],
            "raw_text": raw_text,
            "email": candidate.get("email"),
            "phone": candidate.get("phone")
        }
        results.append(entry)
        
    # Sort descending by final_score
    results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Assign ranks
    for rank_idx, res in enumerate(results, start=1):
        res["rank"] = rank_idx
        
    # Build clean summary dataframe
    df_rows = []
    for r in results:
        df_rows.append({
            "Rank": r["rank"],
            "Candidate": r["candidate_name"],
            "Match Score": f"{r['final_score']:.1f}%",
            "Semantic Match": f"{r['semantic_score']:.1f}%",
            "Skills Matched": f"{len(r['matched_skills'])} / {len(r['matched_skills']) + len(r['missing_skills'])} ({r['skill_match_pct']}%)",
            "Experience": r["experience_status"],
            "Status": r["qualification"],
            "Filename": r["filename"]
        })
        
    ranked_df = pd.DataFrame(df_rows)
    return results, ranked_df
