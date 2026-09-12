"""
app.py
Streamlit Web Application for the AI-Based Resume Screening System.
Provides an interactive recruiter dashboard with semantic matching, candidate ranking,
and Explainable AI (XAI) skill gap breakdowns.
"""

import os
import sys
import pandas as pd
import streamlit as st

# Add parent directory to path so 'src' can be imported cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from src.data_loader import load_job_requirements, list_resume_files
from src.text_extraction import extract_text, extract_metadata
from src.ranking import rank_candidates
from src.embeddings import get_embedding_model


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Resume Screening & Ranking System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F9FAFB, #F3F4F6);
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .skill-tag-matched {
        display: inline-block;
        background-color: #DEF7EC;
        color: #03543F;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        margin: 3px 4px;
        border: 1px solid #84E1BC;
    }
    .skill-tag-missing {
        display: inline-block;
        background-color: #FDE8E8;
        color: #9B1C1C;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        margin: 3px 4px;
        border: 1px solid #F8B4B4;
    }
    .job-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
def load_model():
    """Cache the SentenceTransformer model to avoid repeated downloads/loads."""
    return get_embedding_model()


@st.cache_data
def get_benchmark_jobs():
    """Load job requirements CSV."""
    csv_path = os.path.join(PARENT_DIR, "dataset", "job_requirements.csv")
    if os.path.exists(csv_path):
        return load_job_requirements(csv_path)
    return pd.DataFrame()


def main():
    # Header Banner
    st.markdown('<div class="main-header">🎯 AI-Based Resume Screening System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated Candidate Ranking & Explainable AI Match Analysis powered by Sentence Transformers</div>',
        unsafe_allow_html=True
    )

    # Pre-load transformer model
    model = load_model()

    # Sidebar: Setup & Job Selection
    st.sidebar.header("📋 1. Job Description Setup")
    
    jobs_df = get_benchmark_jobs()
    job_mode = st.sidebar.radio("Job Source", ["Select from Benchmark Dataset", "Enter Custom Job Requirement"])
    
    selected_job = {}
    
    if job_mode == "Select from Benchmark Dataset" and not jobs_df.empty:
        job_options = [f"{row['job_title']} ({row['job_id']})" for _, row in jobs_df.iterrows()]
        selected_option = st.sidebar.selectbox("Choose Benchmark Job", job_options)
        
        # Find matching row
        selected_idx = job_options.index(selected_option)
        selected_job = jobs_df.iloc[selected_idx].to_dict()
    else:
        # Custom Job Requirement Input
        custom_title = st.sidebar.text_input("Job Title", "Machine Learning Specialist")
        custom_category = st.sidebar.text_input("Category", "Artificial Intelligence")
        custom_exp = st.sidebar.text_input("Experience Requirement", "3+ years")
        custom_skills = st.sidebar.text_area("Required Skills (comma separated)", "Python, Machine Learning, PyTorch, SQL, Docker")
        custom_desc = st.sidebar.text_area("Job Description", "We are hiring an ML specialist to develop deep learning models and build scalable data pipelines.")
        
        selected_job = {
            "job_id": "CUSTOM01",
            "job_title": custom_title,
            "category": custom_category,
            "experience_years": custom_exp,
            "required_skills": custom_skills,
            "job_description": custom_desc,
            "education_requirement": "Degree in CS or related",
            "salary_range": "Negotiable",
            "full_job_text": f"Title: {custom_title}. Category: {custom_category}. Skills: {custom_skills}. Experience: {custom_exp}. Description: {custom_desc}"
        }

    # Sidebar: Matching Algorithm Configuration
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 2. Matching Configuration")
    scoring_mode = st.sidebar.selectbox(
        "Scoring Methodology",
        ["Hybrid (Semantic + Skills + Experience)", "Pure Semantic Cosine Similarity"]
    )
    use_hybrid = (scoring_mode.startswith("Hybrid"))
    
    weights = (0.70, 0.20, 0.10)
    if use_hybrid:
        with st.sidebar.expander("Fine-tune Hybrid Weights", expanded=False):
            w_sem = st.slider("Semantic Vector Weight", 0.0, 1.0, 0.70, step=0.05)
            w_skill = st.slider("Skill Match Weight", 0.0, 1.0, 0.20, step=0.05)
            w_exp = st.slider("Experience Weight", 0.0, 1.0, 0.10, step=0.05)
            weights = (w_sem, w_skill, w_exp)
            
    min_score_threshold = st.sidebar.slider("Minimum Qualified Score (%)", 0, 100, 60)

    # Main Section: Display Selected Job Details
    st.markdown("### 📌 Target Job Details")
    st.markdown(f"""
    <div class="job-card">
        <h4 style="margin: 0; color: #1E3A8A;">{selected_job.get('job_title', 'Job')} <span style="font-size: 0.8rem; color: #64748B;">({selected_job.get('job_id', 'ID')})</span></h4>
        <p style="margin: 6px 0; color: #334155;"><strong>Category:</strong> {selected_job.get('category', 'N/A')} | <strong>Required Experience:</strong> {selected_job.get('experience_years', 'N/A')} | <strong>Salary:</strong> {selected_job.get('salary_range', 'N/A')}</p>
        <p style="margin: 6px 0; color: #334155;"><strong>Required Skills:</strong> <code>{selected_job.get('required_skills', 'N/A')}</code></p>
        <p style="margin: 6px 0 0 0; color: #475569; font-size: 0.95rem;">{selected_job.get('job_description', '')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Main Section: Resumes Input
    st.markdown("### 📂 Candidate Resumes")
    resume_source = st.radio(
        "Select Resumes Source",
        ["Use Pre-loaded Sample Resumes (dataset/resumes)", "Upload Candidate Resumes (PDF, DOCX, TXT)"],
        horizontal=True
    )

    candidates = []
    
    if resume_source == "Use Pre-loaded Sample Resumes (dataset/resumes)":
        resumes_dir = os.path.join(PARENT_DIR, "dataset", "resumes")
        sample_files = list_resume_files(resumes_dir)
        
        if not sample_files:
            st.warning("No sample resumes found in `dataset/resumes`. Please run the generator script or upload files.")
        else:
            st.info(f"Loaded {len(sample_files)} sample candidate resumes from `dataset/resumes`.")
            for f in sample_files:
                text = extract_text(f["filepath"], f["filename"])
                meta = extract_metadata(text, f["filename"])
                candidates.append({
                    "filename": f["filename"],
                    "raw_text": text,
                    "candidate_name": meta["candidate_name"],
                    "email": meta["email"],
                    "phone": meta["phone"]
                })
    else:
        uploaded_files = st.file_uploader(
            "Upload Candidate Resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Select one or multiple PDF, Word (.docx), or Text resumes."
        )
        
        if uploaded_files:
            for up_file in uploaded_files:
                text = extract_text(up_file, up_file.name)
                meta = extract_metadata(text, up_file.name)
                candidates.append({
                    "filename": up_file.name,
                    "raw_text": text,
                    "candidate_name": meta["candidate_name"],
                    "email": meta["email"],
                    "phone": meta["phone"]
                })
            st.success(f"Ready to screen {len(candidates)} uploaded resumes!")

    st.write("")
    # Screen Candidates Action
    screen_btn = st.button("🚀 Screen & Rank Candidates", type="primary", use_container_width=True)

    if screen_btn or "screening_results" in st.session_state:
        if screen_btn:
            if not candidates:
                st.error("Please provide at least one candidate resume to screen.")
                return
                
            with st.spinner("AI Screening in progress: extracting features, encoding embeddings, and calculating match scores..."):
                results, ranked_df = rank_candidates(
                    candidates=candidates,
                    job_info=selected_job,
                    use_hybrid=use_hybrid,
                    weights=weights,
                    model=model
                )
                st.session_state["screening_results"] = results
                st.session_state["screening_df"] = ranked_df
                st.session_state["target_job"] = selected_job

        results = st.session_state.get("screening_results", [])
        ranked_df = st.session_state.get("screening_df", pd.DataFrame())

        if results:
            st.markdown("---")
            st.markdown("## 📊 Screening Results & Candidate Rankings")

            # Metrics Row
            total_cand = len(results)
            top_score = results[0]["final_score"]
            avg_score = sum(r["final_score"] for r in results) / total_cand
            qualified_count = sum(1 for r in results if r["final_score"] >= min_score_threshold)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{total_cand}</div><div class="metric-label">Resumes Screened</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{top_score:.1f}%</div><div class="metric-label">Top Match Score</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_score:.1f}%</div><div class="metric-label">Average Match Score</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{qualified_count} / {total_cand}</div><div class="metric-label">Met Threshold (≥{min_score_threshold}%)</div></div>', unsafe_allow_html=True)

            st.write("")
            st.markdown("### 🏆 Candidate Leaderboard")
            
            # Display Leaderboard Table
            st.dataframe(
                ranked_df,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Candidate": st.column_config.TextColumn("Candidate Name"),
                    "Match Score": st.column_config.TextColumn("Final Score"),
                    "Semantic Match": st.column_config.TextColumn("Semantic Match"),
                    "Skills Matched": st.column_config.TextColumn("Skills Overview"),
                    "Experience": st.column_config.TextColumn("Experience Fit"),
                    "Status": st.column_config.TextColumn("Status"),
                    "Filename": st.column_config.TextColumn("Source File")
                },
                use_container_width=True,
                hide_index=True
            )

            # Export Results Button
            csv_export = ranked_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Screening Report (CSV)",
                data=csv_export,
                file_name=f"screening_report_{selected_job.get('job_id', 'job')}.csv",
                mime="text/csv"
            )

            # Candidate Explainable AI (XAI) Deep Dive
            st.markdown("---")
            st.markdown("## 🔍 Explainable AI (XAI) Candidate Deep Dive")
            st.write("Inspect why a candidate was ranked at their position, including matched skills, missing skills, and experience gap analysis.")

            candidate_names = [f"Rank {r['rank']}: {r['candidate_name']} ({r['final_score']:.1f}%)" for r in results]
            selected_cand_idx = st.selectbox("Select Candidate to Inspect", range(len(candidate_names)), format_func=lambda x: candidate_names[x])

            cand_detail = results[selected_cand_idx]

            # Candidate Profile Summary
            c_col1, c_col2 = st.columns([1, 2])
            with c_col1:
                st.markdown(f"### {cand_detail['candidate_name']}")
                st.write(f"📄 **File:** `{cand_detail['filename']}`")
                if cand_detail.get("email"):
                    st.write(f"📧 **Email:** {cand_detail['email']}")
                if cand_detail.get("phone"):
                    st.write(f"📞 **Phone:** {cand_detail['phone']}")
                st.write(f"🏷️ **Status:** **{cand_detail['qualification']}**")
                
                # Match score gauge/progress
                st.metric("Overall Match Score", f"{cand_detail['final_score']:.1f}%")
                st.progress(min(1.0, cand_detail['final_score'] / 100.0))
                
                st.metric("Pure Semantic Cosine Match", f"{cand_detail['semantic_score']:.1f}%")
                st.metric("Skills Overlap Ratio", f"{cand_detail['skill_match_pct']:.1f}%")

            with c_col2:
                st.markdown("#### 💡 AI Match Justification")
                st.info(cand_detail["summary"])

                # Matched Skills
                st.markdown("#### ✅ Matched Skills")
                if cand_detail["matched_skills"]:
                    tags_html = "".join([f'<span class="skill-tag-matched">✓ {skill}</span>' for skill in cand_detail["matched_skills"]])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.write("*No direct skill matches found in resume text.*")

                # Missing Skills
                st.markdown("#### ❌ Missing / Unverified Skills")
                if cand_detail["missing_skills"]:
                    tags_html = "".join([f'<span class="skill-tag-missing">✗ {skill}</span>' for skill in cand_detail["missing_skills"]])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.write("*Candidate fulfills all listed technical skills!*")

                # Experience Fit
                st.markdown("#### ⏳ Experience Analysis")
                st.write(f"- **Candidate Detected Experience:** `{cand_detail['candidate_exp_years'] if cand_detail['candidate_exp_years'] is not None else 'Not explicitly detected'} years`")
                st.write(f"- **Job Required Experience:** `{selected_job.get('experience_years', 'N/A')}`")
                st.write(f"- **Status:** {cand_detail['experience_status']}")

            # Extracted Resume Text Accordion
            with st.expander("📄 View Extracted Resume Raw Text"):
                st.text(cand_detail.get("raw_text", "No text available"))


if __name__ == "__main__":
    main()
