"""
data_loader.py
Module for loading job requirements dataset and discovering candidate resume files.
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd

REQUIRED_COLUMNS = [
    "job_id",
    "job_title",
    "category",
    "education_requirement",
    "experience_years",
    "required_skills",
    "job_description",
    "salary_range",
]


def load_job_requirements(filepath: str = "dataset/job_requirements.csv") -> pd.DataFrame:
    """
    Load job requirements from a CSV file and validate/clean the schema.
    
    Args:
        filepath: Path to the job requirements CSV.
        
    Returns:
        pd.DataFrame containing sanitized job requirements.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Job requirements file not found at: {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Check for missing required columns and fill if missing
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
            
    # Clean string fields
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str).str.strip()
            
    # Construct combined search/embedding text for each job
    df["full_job_text"] = df.apply(
        lambda row: (
            f"Job Title: {row['job_title']}. "
            f"Category: {row['category']}. "
            f"Required Skills: {row['required_skills']}. "
            f"Experience Required: {row['experience_years']}. "
            f"Education: {row['education_requirement']}. "
            f"Job Description: {row['job_description']}"
        ),
        axis=1
    )
    
    return df


def get_job_by_id(df: pd.DataFrame, job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single job dictionary by its job_id."""
    match = df[df["job_id"].astype(str) == str(job_id)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def list_resume_files(directory_path: str = "dataset/resumes") -> List[Dict[str, str]]:
    """
    Scan a directory for supported resume documents (.pdf, .docx, .txt).
    
    Args:
        directory_path: Path to resumes folder.
        
    Returns:
        List of dicts with keys: 'filename', 'filepath', 'extension'.
    """
    if not os.path.exists(directory_path):
        return []
        
    supported_exts = {".pdf", ".docx", ".txt"}
    resumes = []
    
    for fname in sorted(os.listdir(directory_path)):
        _, ext = os.path.splitext(fname)
        if ext.lower() in supported_exts:
            resumes.append({
                "filename": fname,
                "filepath": os.path.join(directory_path, fname),
                "extension": ext.lower()
            })
            
    return resumes
