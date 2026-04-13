import streamlit as st
import time
from nlp_engine import ResumeScreener

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0d0f12; }
    .stApp { background-color: #0d0f12; color: #e8eaed; }

    .header-box {
        background: #161a1f;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .header-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 22px;
        font-weight: 600;
        color: #e8eaed;
    }
    .header-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #7a8494;
        margin-top: 2px;
    }

    .score-card {
        background: #161a1f;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.07);
        padding: 28px;
        text-align: center;
    }
    .score-big {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 72px;
        font-weight: 600;
        line-height: 1;
        margin-bottom: 4px;
    }
    .score-label {
        font-size: 18px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .score-desc {
        font-size: 13px;
        color: #7a8494;
        line-height: 1.5;
    }

    .metric-card {
        background: #1e2329;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 32px;
        font-weight: 600;
    }
    .metric-lbl {
        font-size: 12px;
        color: #7a8494;
        margin-top: 2px;
    }

    .kw-matched { background-color: rgba(0,153,255,0.12); color: #0099ff;
        border: 1px solid rgba(0,153,255,0.25); border-radius: 20px;
        padding: 3px 12px; font-size: 12px; font-family: 'IBM Plex Mono', monospace;
        display: inline-block; margin: 3px; }
    .kw-missing { background-color: rgba(255,107,107,0.1); color: #ff6b6b;
        border: 1px solid rgba(255,107,107,0.25); border-radius: 20px;
        padding: 3px 12px; font-size: 12px; font-family: 'IBM Plex Mono', monospace;
        display: inline-block; margin: 3px; }
    .kw-bonus  { background-color: rgba(0,229,160,0.08); color: #00e5a0;
        border: 1px solid rgba(0,229,160,0.2); border-radius: 20px;
        padding: 3px 12px; font-size: 12px; font-family: 'IBM Plex Mono', monospace;
        display: inline-block; margin: 3px; }

    .panel {
        background: #161a1f;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .panel-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }
    .sug-item {
        border-left: 3px solid #f0a500;
        background: rgba(240,165,0,0.06);
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 13px;
        line-height: 1.6;
        color: #e8eaed;
    }
    .stTextArea textarea {
        background: #1e2329 !important;
        color: #e8eaed !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 13px !important;
    }
    .stButton>button {
        background: #00e5a0 !important;
        color: #000 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-size: 14px !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    .stButton>button:hover { background: #00ffb3 !important; }
    div[data-testid="stProgress"] > div > div {
        background: #00e5a0 !important;
    }
    .stSelectbox select, .stSelectbox div {
        background: #1e2329 !important;
        color: #e8eaed !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div style="background:#00e5a0;border-radius:10px;width:44px;height:44px;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;">📄</div>
    <div>
        <div class="header-title">Resume.AI Screener</div>
        <div class="header-sub">TF-IDF Vectorization · Cosine Similarity · NLP · ATS Analysis</div>
    </div>
    <div style="margin-left:auto;font-family:'IBM Plex Mono',monospace;font-size:11px;
        background:rgba(0,229,160,0.1);color:#00e5a0;border:1px solid rgba(0,229,160,0.2);
        padding:6px 14px;border-radius:20px;">v2.0 LIVE</div>
</div>
""", unsafe_allow_html=True)

# ── Presets ───────────────────────────────────────────────────────────────────
PRESETS_JD = {
    "None": "",
    "ML Engineer": """We are looking for a Senior Machine Learning Engineer with 4+ years of experience.
Required: Python, TensorFlow, PyTorch, scikit-learn, NLP, computer vision, Docker, Kubernetes, AWS, REST APIs, SQL, data pipelines, MLOps, Git, team collaboration.
Nice to have: Spark, Kafka, Airflow, experience with LLMs and transformer models.""",
    "Frontend Developer": """Frontend Developer with strong React.js experience. Must know JavaScript ES6, TypeScript, HTML5, CSS3, Redux, REST APIs, GraphQL, responsive design, Webpack, Git, CI/CD. Experience with Next.js, testing with Jest and Cypress, accessibility WCAG, and Figma preferred.""",
    "Data Analyst": """Data Analyst needed with skills in SQL, Python, Excel, Tableau, Power BI, data visualization, statistics, pandas, NumPy, data cleaning, A/B testing, Google Analytics, stakeholder reporting, and business intelligence.""",
}

PRESETS_RESUME = {
    "None": "",
    "Strong Match": """Senior Python Engineer — 5 years experience.
Skills: Python, TensorFlow, PyTorch, scikit-learn, NLP pipelines, Docker, Kubernetes, AWS, REST API development, PostgreSQL, Git, MLOps, Airflow, Spark.
Built transformer-based NLP models. Deployed ML pipelines on AWS using Kubernetes. Cross-functional collaborator.""",
    "Partial Match": """Software Developer — 3 years.
Skills: Python, scikit-learn, pandas, REST APIs, PostgreSQL, Git, Linux.
Some machine learning experience. Worked in agile teams. Built data processing scripts and basic ML models. No cloud or container experience yet.""",
    "Weak Match": """Recent Graduate — Computer Science B.Sc.
Courses: Java, C++, basic Python, data structures. Projects: calculator app, grade management system.
Internship: IT support 2 months. Microsoft Office proficient. Eager to learn.""",
}

# ── Input Section ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📋 Job Description")
    jd_preset = st.selectbox("Load preset JD", list(PRESETS_JD.keys()), key="jd_preset")
    jd_text = st.text_area(
        "Paste the job description",
        value=PRESETS_JD[jd_preset],
        height=260,
        placeholder="Paste job description here...",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("#### 👤 Candidate Resume")
    resume_preset = st.selectbox("Load preset resume", list(PRESETS_RESUME.keys()), key="res_preset")
    resume_text = st.text_area(
        "Paste the resume",
        value=PRESETS_RESUME[resume_preset],
        height=260,
        placeholder="Paste resume text here...",
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("⚡  Analyze Resume Match", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if run:
    if not jd_text.strip() or not resume_text.strip():
        st.warning("⚠️  Please fill in both the Job Description and Resume fields.")
    else:
        with st.spinner("Running NLP analysis..."):
            time.sleep(0.3)
            screener = ResumeScreener()
            result = screener.analyze(jd_text, resume_text)

        score = result["score"]
        color = "#00e5a0" if score >= 70 else "#f0a500" if score >= 45 else "#ff6b6b"

        # Score section
        if score >= 80:
            label, desc = "🟢 Excellent Match", "This resume strongly aligns with the job. Highly likely to pass ATS screening."
        elif score >= 60:
            label, desc = "🟡 Good Match", "Solid alignment with most requirements. A few tweaks could make this much stronger."
        elif score >= 40:
            label, desc = "🟠 Partial Match", "Some relevant skills present, but important keywords are missing."
        else:
            label, desc = "🔴 Weak Match", "Resume lacks most critical keywords. Major rework recommended before applying."

        st.markdown("---")
        st.markdown("### 📊 Analysis Results")

        # Score + mini stats
        sc, m1, m2, m3 = st.columns([2, 1, 1, 1])
        with sc:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-big" style="color:{color}">{score}%</div>
                <div class="score-label">{label}</div>
                <div class="score-desc">{desc}</div>
                <div style="margin-top:16px;">
                    <div style="background:#1e2329;border-radius:8px;height:8px;overflow:hidden;">
                        <div style="width:{score}%;height:100%;background:{color};border-radius:8px;transition:width 1s;"></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val" style="color:#0099ff">{len(result['matched'])}</div>
                <div class="metric-lbl">Keywords matched</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val" style="color:#ff6b6b">{len(result['missing'])}</div>
                <div class="metric-lbl">Keywords missing</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-val" style="color:#00e5a0">{len(result['bonus'])}</div>
                <div class="metric-lbl">Bonus skills</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Keywords
        kc1, kc2 = st.columns(2)
        with kc1:
            st.markdown('<div class="panel"><div class="panel-title" style="color:#0099ff">✦ Matched Keywords</div>', unsafe_allow_html=True)
            if result["matched"]:
                st.markdown(" ".join(f'<span class="kw-matched">{k}</span>' for k in result["matched"]), unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#7a8494;font-size:13px;">No matching keywords found</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with kc2:
            st.markdown('<div class="panel"><div class="panel-title" style="color:#ff6b6b">✦ Missing Keywords</div>', unsafe_allow_html=True)
            if result["missing"]:
                st.markdown(" ".join(f'<span class="kw-missing">{k}</span>' for k in result["missing"]), unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#00e5a0;font-size:13px;">✓ All keywords matched!</span>', unsafe_allow_html=True)
            if result["bonus"]:
                st.markdown('<div style="margin-top:12px;font-family:IBM Plex Mono,monospace;font-size:11px;color:#7a8494;margin-bottom:6px;">BONUS SKILLS</div>', unsafe_allow_html=True)
                st.markdown(" ".join(f'<span class="kw-bonus">{k}</span>' for k in result["bonus"][:8]), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # TF-IDF bar chart
        st.markdown('<div class="panel"><div class="panel-title" style="color:#7a8494">✦ Top TF-IDF Terms — Job Description</div>', unsafe_allow_html=True)
        if result["top_tfidf"]:
            max_val = result["top_tfidf"][0][1]
            for term, val in result["top_tfidf"]:
                pct = int((val / max_val) * 100) if max_val else 0
                bar_color = "#00e5a0" if term in result["matched"] else "#378add"
                cols = st.columns([2, 6, 1])
                cols[0].markdown(f'<span style="font-family:IBM Plex Mono,monospace;font-size:12px;">{term}</span>', unsafe_allow_html=True)
                cols[1].markdown(f'<div style="background:#1e2329;border-radius:3px;height:10px;margin-top:6px;overflow:hidden;"><div style="width:{pct}%;height:100%;background:{bar_color};border-radius:3px;"></div></div>', unsafe_allow_html=True)
                cols[2].markdown(f'<span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#7a8494;">{val:.2f}</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Suggestions
        st.markdown('<div class="panel"><div class="panel-title" style="color:#f0a500">✦ Improvement Suggestions</div>', unsafe_allow_html=True)
        for tip in result["suggestions"]:
            st.markdown(f'<div class="sug-item">{tip}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Download report
        report = screener.generate_report(result, jd_text, resume_text)
        st.download_button(
            label="📥 Download Full Report (.txt)",
            data=report,
            file_name="resume_screening_report.txt",
            mime="text/plain"
        )
