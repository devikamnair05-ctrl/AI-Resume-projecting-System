# 🎯 AI-Based Resume Screening System

An automated, intelligent Resume Screening and Candidate Ranking platform built using **Sentence Transformers (`all-MiniLM-L6-v2`)**, **PyMuPDF**, and **Streamlit**. 

The system takes job requirements and candidate resumes (PDF, DOCX, or TXT), computes dense semantic vector embeddings, calculates cosine similarity match scores, ranks candidates on a leaderboard, and provides **Explainable AI (XAI)** breakdowns detailing matched skills, missing skills, and experience fit.

---

## 🌟 Key Features

- **Multi-Format Text Extraction**: Extracts text and metadata (name, email, phone) from PDF (`PyMuPDF`), Word (`python-docx`), and plain text resumes.
- **NLP Preprocessing**: Normalizes text while preserving technical keywords (e.g., `C++`, `C#`, `.NET`, `CI/CD`, `Node.js`, `Scikit-Learn`).
- **Transformer Embeddings**: Employs `all-MiniLM-L6-v2` from Sentence Transformers for 384-dimensional dense semantic representations.
- **Dual Matching Modes**:
  - **Pure Semantic Similarity**: Direct cosine similarity between job embedding and resume embedding.
  - **Explainable Hybrid Scoring**: Weighted combination of semantic similarity, exact/fuzzy skill overlap, and experience fit.
- **Explainable AI (XAI)**:
  - **Matched Skills**: Highlights technical skills present in both the JD and the candidate's resume (✓).
  - **Missing Skills**: Clearly flags required skills not found in the resume (✗).
  - **Experience Analysis**: Compares candidate's estimated years of experience against JD requirements.
- **Interactive Recruiter Dashboard (Streamlit)**:
  - Select from pre-loaded benchmark jobs or input custom requirements.
  - Test with pre-generated sample resumes or upload multiple PDF/Word resumes.
  - View real-time leaderboards, candidate metrics, and export screening results to CSV.

---

## 📁 Project Folder Structure

```
AI_Resume_Screening/
│
├── dataset/
│   ├── job_requirements.csv          # Benchmark job postings
│   └── resumes/                      # Candidate PDF resumes
│       ├── resume1_anu_ml_senior.pdf
│       ├── resume2_meera_ds_mid.pdf
│       ├── resume3_rahul_devops.pdf
│       ├── resume4_vikram_fullstack.pdf
│       ├── resume5_priya_frontend.pdf
│       └── resume6_arjun_marketing_intern.pdf
│
├── src/                              # Core Python pipeline modules
│   ├── __init__.py
│   ├── data_loader.py                # Load CSV and scan resume folders
│   ├── text_extraction.py            # PDF/DOCX text & metadata extraction
│   ├── preprocessing.py              # Text cleaning & tech-term preservation
│   ├── embeddings.py                 # Sentence Transformer model loader
│   ├── matching.py                   # Cosine similarity & hybrid scoring
│   ├── ranking.py                    # Candidate ranking & DataFrame builder
│   └── explainability.py             # XAI skill matching & justification
│
├── models/                           # Local cache for model weights
│   └── .gitkeep
│
├── app/
│   └── app.py                        # Streamlit recruiter dashboard
│
├── notebooks/
│   └── experiments.ipynb             # Step-by-step pipeline experiments
│
├── scripts/
│   └── generate_sample_resumes.py    # Synthetic PDF resume generator
│
├── tests/
│   └── test_pipeline.py              # Automated pipeline test suite
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Generate Sample PDF Resumes (Optional)
Sample resumes are already included, but you can regenerate them anytime:
```powershell
python scripts/generate_sample_resumes.py
```

### 4. Run Automated Pipeline Verification Test
```powershell
python tests/test_pipeline.py
```

### 5. Launch the Streamlit Recruiter Dashboard
```powershell
streamlit run app/app.py
```

---

## 🔄 Processing Pipeline

```
               INPUT
                 │
       ┌─────────┴──────────┐
       │                    │
   Resumes           Job Requirements
 (PDF/DOCX)               (CSV)
       │                    │
       ▼                    ▼
 Text Extraction        Job Text
       │                    │
       └─────────┬──────────┘
                 ▼
          NLP Preprocessing
                 │
                 ▼
        Sentence Transformer
         (all-MiniLM-L6-v2)
                 │
                 ▼
            Embeddings
                 │
                 ▼
         Cosine Similarity
                 │
                 ▼
         Candidate Ranking
                 │
                 ▼
        Explainable AI (XAI)
     (Matched vs Missing Skills)
                 │
                 ▼
      Interactive UI Dashboard
```

---

## 🧩 Pipeline Modules Explained

| Module | Primary Responsibilities |
|---|---|
| `data_loader.py` | Validates and loads `job_requirements.csv`, structures composite job texts, and discovers resume files. |
| `text_extraction.py` | Extracts text from PDF files using `PyMuPDF` (`fitz`), `.docx` via `python-docx`, and extracts candidate name/email heuristics. |
| `preprocessing.py` | Cleans raw text, protects technical keywords (`C++`, `.NET`, `CI/CD`), and parses stated years of experience. |
| `embeddings.py` | Cached loader for `all-MiniLM-L6-v2` producing normalized 384-dimensional dense vectors. |
| `matching.py` | Vector cosine similarity calculation and optional weighted hybrid scoring. |
| `ranking.py` | Computes match metrics across all candidates, sorts descending, and formats leaderboards. |
| `explainability.py` | Analyzes exact and fuzzy skill overlap, experience fit, and produces qualitative candidate justifications. |
