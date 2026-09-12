"""
preprocessing.py
Module for text cleaning, normalization, and NLP feature extraction.
Preserves technical terms (e.g., C++, C#, .NET, CI/CD, Node.js) while removing noise.
"""

import re
from typing import List, Optional


# Protected tokens that shouldn't have special characters stripped
TECH_TOKENS_MAP = {
    "c++": "cplusplus",
    "c#": "csharp",
    ".net": "dotnet",
    "ci/cd": "cicd",
    "node.js": "nodejs",
    "vue.js": "vuejs",
    "next.js": "nextjs",
    "scikit-learn": "scikitlearn",
    "scikit learn": "scikitlearn",
}

REVERSE_TECH_TOKENS = {v: k for k, v in TECH_TOKENS_MAP.items()}


def clean_text(text: str, preserve_tech_terms: bool = True) -> str:
    """
    Clean and normalize resume or job description text.
    
    Operations:
    1. Lowercase
    2. Protect critical technical terms
    3. Remove email and url strings
    4. Remove noise punctuation while keeping whitespace
    5. Collapse multiple spaces
    
    Args:
        text: Raw input text
        preserve_tech_terms: Whether to map tech tokens before punctuation stripping
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
        
    cleaned = text.lower()
    
    if preserve_tech_terms:
        for tech, replacement in TECH_TOKENS_MAP.items():
            cleaned = re.sub(r'\b' + re.escape(tech) + r'\b', replacement, cleaned)
            
    # Remove URLs
    cleaned = re.sub(r'http\S+|www\.\S+', ' ', cleaned)
    
    # Remove email addresses
    cleaned = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', ' ', cleaned)
    
    # Replace punctuation (except alphanumeric and spaces) with spaces
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    
    # Collapse multiple whitespace characters into single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def extract_skills_list(skills_input: str) -> List[str]:
    """
    Parse a comma, bullet, or newline separated skill string into a standardized list.
    
    Args:
        skills_input: e.g. "Python, Machine Learning, Deep Learning, SQL"
        
    Returns:
        List of distinct stripped skill strings.
    """
    if not skills_input:
        return []
        
    # Split on commas, semicolons, or slashes
    raw_tokens = re.split(r'[,;|\n•·\t]+', skills_input)
    skills = []
    
    for token in raw_tokens:
        clean_token = token.strip()
        # Remove leading/trailing bullet symbols
        clean_token = re.sub(r'^[-*•\s]+', '', clean_token).strip()
        if clean_token and len(clean_token) > 1 and clean_token not in skills:
            skills.append(clean_token)
            
    return skills


def extract_years_of_experience(text: str) -> Optional[float]:
    """
    Extract estimated years of experience from resume text using regex heuristics.
    
    Looks for phrases such as:
    - '4 years of experience'
    - 'over 3.5 years'
    - '2+ years'
    - '(2020 - 2024)'
    
    Returns:
        float representing years of experience, or None if not determinable.
    """
    if not text:
        return None
        
    text_lower = text.lower()
    
    # Pattern 1: Direct statements like "4 years of experience", "2+ years", "3.5 yrs"
    exp_pattern = r'(\d+(?:\.\d+)?)\s*(?:\+|-\d+)?\s*(?:years?|yrs?)(?:\s+(?:of\s+)?experience)?'
    matches = re.findall(exp_pattern, text_lower)
    
    found_years = []
    if matches:
        for m in matches:
            try:
                val = float(m)
                # Realistic bound for a single mention of experience: 0.5 to 35
                if 0.5 <= val <= 35:
                    found_years.append(val)
            except ValueError:
                continue
                
    if found_years:
        # Return the maximum explicit mention (often the candidate's total experience in summary)
        return max(found_years)
        
    # Pattern 2: Date ranges like 2020 - 2024 or 2019 - Present
    date_range_pattern = r'(20\d\d)\s*(?:-|–|to)\s*(20\d\d|present)'
    date_matches = re.findall(date_range_pattern, text_lower)
    
    current_year = 2024
    total_span = 0.0
    for start_str, end_str in date_matches:
        try:
            start_yr = int(start_str)
            end_yr = current_year if "present" in end_str else int(end_str)
            span = max(0, end_yr - start_yr)
            if span > total_span:
                total_span = span
        except ValueError:
            continue
            
    return total_span if total_span > 0 else None
