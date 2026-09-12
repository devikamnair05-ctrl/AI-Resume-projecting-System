"""
explainability.py
Module for Explainable AI (XAI) in resume screening.
Breaks down candidate evaluations into matched skills, missing skills,
experience gap analysis, and plain-English justification summaries.
"""

import re
from typing import Dict, List, Any, Optional
from src.preprocessing import extract_skills_list, extract_years_of_experience, clean_text


def check_skill_in_text(skill: str, text: str) -> bool:
    """
    Check whether a specific skill exists in resume text using word boundary aware regex.
    Handles special terms like C++, C#, .NET, Node.js gracefully.
    """
    skill_clean = skill.strip().lower()
    text_clean = text.lower()
    
    if not skill_clean:
        return False
        
    # Exact phrase / word boundary match
    # Special character escaping
    escaped_skill = re.escape(skill_clean)
    
    # If skill starts or ends with symbols like +, #, . word boundary \b may fail in regex
    pattern = rf'(?:^|[\s,;./()\[\]"\'\-])({escaped_skill})(?:$|[\s,;./()\[\]"\'\-])'
    
    if re.search(pattern, text_clean):
        return True
        
    # Secondary check: Normalized token check (e.g., scikit-learn vs scikitlearn)
    norm_skill = re.sub(r'[\s\-_.]', '', skill_clean)
    norm_text = re.sub(r'[\s\-_.]', '', text_clean)
    if len(norm_skill) >= 3 and norm_skill in norm_text:
        return True
        
    return False


def parse_required_experience_years(exp_str: str) -> float:
    """
    Parse required experience from string like '3+', '2-4 years', '5' into a numeric float.
    """
    if not exp_str:
        return 0.0
    match = re.search(r'(\d+(?:\.\d+)?)', str(exp_str))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.0


def explain_candidate_match(
    candidate_text: str,
    job_info: Dict[str, Any],
    semantic_score_pct: float
) -> Dict[str, Any]:
    """
    Generate comprehensive Explainable AI (XAI) analysis for a candidate against a job requirement.
    
    Args:
        candidate_text: Extracted raw or cleaned text of the resume.
        job_info: Dictionary containing job requirements (e.g. required_skills, experience_years, job_title).
        semantic_score_pct: The cosine semantic match score as a percentage (0 - 100%).
        
    Returns:
        Dictionary with XAI breakdown:
        - matched_skills: list of matched skills
        - missing_skills: list of missing skills
        - skill_match_ratio: float (0.0 to 1.0)
        - candidate_exp_years: float or None
        - required_exp_years: float
        - experience_status: str ("Met", "Partially Met", "Not Met", "Unknown")
        - qualification_category: str ("Highly Relevant", "Moderate Match", "Low Match")
        - summary: str
    """
    # 1. Skills Breakdown
    raw_required_skills = job_info.get("required_skills", "")
    skills_list = extract_skills_list(raw_required_skills)
    
    matched_skills = []
    missing_skills = []
    
    for skill in skills_list:
        if check_skill_in_text(skill, candidate_text):
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)
            
    total_skills = len(skills_list)
    skill_match_ratio = (len(matched_skills) / total_skills) if total_skills > 0 else 1.0
    
    # 2. Experience Analysis
    req_exp_str = str(job_info.get("experience_years", "0"))
    required_exp_years = parse_required_experience_years(req_exp_str)
    candidate_exp_years = extract_years_of_experience(candidate_text)
    
    if candidate_exp_years is None:
        exp_status = "Not explicitly stated"
        exp_ratio = 0.8  # Neutral fallback
    elif candidate_exp_years >= required_exp_years:
        exp_status = f"Requirement Met ({candidate_exp_years} yrs vs {required_exp_years}+ yrs req.)"
        exp_ratio = 1.0
    elif candidate_exp_years >= (required_exp_years * 0.7):
        exp_status = f"Close to Requirement ({candidate_exp_years} yrs vs {required_exp_years}+ yrs req.)"
        exp_ratio = candidate_exp_years / required_exp_years
    else:
        exp_status = f"Below Requirement ({candidate_exp_years} yrs vs {required_exp_years}+ yrs req.)"
        exp_ratio = candidate_exp_years / required_exp_years
        
    # 3. Qualification Tier
    if semantic_score_pct >= 80.0:
        qualification = "Highly Relevant"
        qualification_badge = "success"
    elif semantic_score_pct >= 60.0:
        qualification = "Moderate Match"
        qualification_badge = "warning"
    else:
        qualification = "Low Match"
        qualification_badge = "danger"
        
    # 4. Summary Text
    job_title = job_info.get("job_title", "target role")
    summary = (
        f"Candidate scored {semantic_score_pct:.1f}% semantic match for '{job_title}'. "
        f"Matched {len(matched_skills)} of {total_skills} required skills ({skill_match_ratio * 100:.0f}%). "
        f"Experience status: {exp_status}."
    )
    
    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_match_ratio": round(skill_match_ratio, 2),
        "candidate_exp_years": candidate_exp_years,
        "required_exp_years": required_exp_years,
        "experience_status": exp_status,
        "experience_ratio": round(exp_ratio, 2),
        "qualification_category": qualification,
        "qualification_badge": qualification_badge,
        "summary": summary,
    }
