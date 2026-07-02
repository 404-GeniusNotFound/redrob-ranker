"""
Redrob Ranker — Interactive Sandbox

A Streamlit application that demonstrates the full candidate ranking pipeline
with an interactive dashboard. Judges can:
  1. Upload a JD or use the default
  2. See real-time pipeline stages
  3. Explore top 100 rankings with drill-downs
  4. Understand scoring dimensions via charts
  5. Inspect honeypot detection reasoning
"""

import streamlit as st
import pandas as pd
import json
import csv
import time
import sys
import os
from pathlib import Path

# Add parent dir to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    WEIGHTS, MAX_POINTS, MUST_HAVE_SKILL_CONCEPTS,
    NICE_TO_HAVE_SKILL_CONCEPTS, PREFERRED_LOCATIONS, WELCOME_LOCATIONS,
)
from src.filters import apply_hard_filters
from src.jd_parser import build_jd_vectorizer, build_candidate_text, compute_jd_similarity
from src.scorer import compute_candidate_score
from src.honeypot import detect_honeypot
from src.reasoning import generate_reasoning


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob AI Ranker",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global styling */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Main title gradient */
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }

    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #a0aec0;
        font-size: 0.85rem;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Stage indicator */
    .stage-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .stage-complete {
        background: rgba(72, 187, 120, 0.2);
        color: #48bb78;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
    .stage-active {
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }

    /* Candidate cards */
    .candidate-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }

    /* Score bar */
    .score-bar-bg {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        height: 24px;
        position: relative;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    div[data-testid="stSidebarContent"] {
        background: rgba(15, 12, 41, 0.95);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────────────────────

def load_candidates(filepath):
    """Load candidates from JSON or JSONL file."""
    path = Path(filepath)
    if path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    else:
        candidates = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
        return candidates


def render_metric_card(value, label, col):
    """Render a glassmorphic metric card."""
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def score_color(score, max_score):
    """Return a color based on score percentage."""
    pct = score / max_score if max_score > 0 else 0
    if pct >= 0.7:
        return "#48bb78"  # green
    elif pct >= 0.4:
        return "#ecc94b"  # yellow
    else:
        return "#fc8181"  # red


def render_score_bar(label, score, max_score, weight):
    """Render a score dimension bar."""
    pct = (score / max_score * 100) if max_score > 0 else 0
    color = score_color(score, max_score)
    st.markdown(f"""
    <div style="margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #e2e8f0; font-size: 0.85rem;">{label}</span>
            <span style="color: {color}; font-weight: 600; font-size: 0.85rem;">
                {score:.1f}/{max_score:.0f} (w={weight:.0%})
            </span>
        </div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width: {pct:.1f}%; background: linear-gradient(90deg, {color}88, {color});"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Pipeline Controls")

    # Data source selection
    data_option = st.radio(
        "Dataset",
        ["Sample (50 candidates)", "Full (100K candidates)"],
        index=0,
        help="Choose the dataset to run the pipeline on"
    )

    # Find data files
    # Sample path is local to the repo so it works on Streamlit Cloud
    sample_path = Path(__file__).parent / "data" / "sample_candidates.json"
    # Full path might only exist locally during development
    full_path = Path(__file__).parent.parent / "India_runs_data_and_ai_challenge" / "candidates.jsonl"

    if data_option == "Full (100K candidates)":
        if full_path.exists():
            data_path = str(full_path)
            st.info("⚡ Full dataset: ~68s runtime")
        else:
            st.warning("Full dataset not found. Using sample.")
            data_path = str(sample_path)
    else:
        data_path = str(sample_path)

    st.markdown("---")

    # Weight tuning
    st.markdown("### ⚖️ Scoring Weights")
    st.caption("Adjust dimension importance (automatically normalized)")

    w_career = st.slider("Career Coherence", 0, 100, int(WEIGHTS["career_coherence"] * 100), key="w_career")
    w_skill = st.slider("Skill Authenticity", 0, 100, int(WEIGHTS["skill_authenticity"] * 100), key="w_skill")
    w_jd = st.slider("JD Alignment", 0, 100, int(WEIGHTS["jd_semantic_align"] * 100), key="w_jd")
    w_behav = st.slider("Behavioral Availability", 0, 100, int(WEIGHTS["behavioral_avail"] * 100), key="w_behav")
    w_loc = st.slider("Location Fit", 0, 100, int(WEIGHTS["location_fit"] * 100), key="w_loc")
    w_edu = st.slider("Education & Credibility", 0, 100, int(WEIGHTS["education_cred"] * 100), key="w_edu")

    # Normalize
    total_w = w_career + w_skill + w_jd + w_behav + w_loc + w_edu
    if total_w > 0:
        custom_weights = {
            "career_coherence": w_career / total_w,
            "skill_authenticity": w_skill / total_w,
            "jd_semantic_align": w_jd / total_w,
            "behavioral_avail": w_behav / total_w,
            "location_fit": w_loc / total_w,
            "education_cred": w_edu / total_w,
        }
    else:
        custom_weights = WEIGHTS.copy()

    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    Built by **Sathvik V** for the Redrob Hackathon.

    **Architecture**: 5-stage pipeline with 6 scoring dimensions,
    honeypot detection, and templated reasoning.

    **Runtime**: ~68s on 100K candidates (CPU-only, no GPU).
    """)

    run_button = st.button("🚀 Run Pipeline", use_container_width=True, type="primary")


# ── Main Content ─────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🏆 Redrob AI Ranker</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent Candidate Discovery & Ranking for Senior AI Engineer</p>', unsafe_allow_html=True)


# ── Pipeline Execution ───────────────────────────────────────────────────────
if run_button or "results" in st.session_state:

    if run_button:
        # Run the pipeline
        progress_bar = st.progress(0, "Loading candidates...")

        # Stage 0: Load
        candidates = load_candidates(data_path)
        total = len(candidates)
        progress_bar.progress(5, f"Loaded {total:,} candidates")

        # Stage 1: Filter
        progress_bar.progress(10, "Stage 1: Applying hard filters...")
        filtered = [c for c in candidates if apply_hard_filters(c)]
        n_filtered = len(filtered)

        # Stage 2-3: Score
        progress_bar.progress(20, "Stage 2-3: Scoring candidates...")
        vectorizer, jd_vector = build_jd_vectorizer()

        scored = []
        for i, c in enumerate(filtered):
            cand_text = build_candidate_text(c)
            jd_sim = compute_jd_similarity(cand_text, vectorizer, jd_vector)
            scores = compute_candidate_score(c, jd_sim)

            # Apply custom weights
            final = sum(
                (scores[dim] / MAX_POINTS[dim]) * custom_weights[dim]
                for dim in WEIGHTS.keys()
            )
            scores["final_score"] = round(final, 6)

            scored.append({"candidate": c, "scores": scores, "final_score": final})

            if (i + 1) % 500 == 0:
                pct = 20 + int(60 * (i + 1) / n_filtered)
                progress_bar.progress(min(pct, 80), f"Scored {i+1:,}/{n_filtered:,}...")

        # Stage 4: Honeypot
        progress_bar.progress(85, "Stage 4: Detecting honeypots...")
        honeypot_count = 0
        for sc in scored:
            is_hp, reasons = detect_honeypot(sc["candidate"])
            sc["is_honeypot"] = is_hp
            sc["honeypot_reasons"] = reasons
            if is_hp:
                honeypot_count += 1

        # Sort and select top 100
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top_100 = [sc for sc in scored if not sc["is_honeypot"]][:100]

        # Stage 5: Reasoning
        progress_bar.progress(95, "Stage 5: Generating reasoning...")
        results = []
        for rank_idx, sc in enumerate(top_100):
            rank = rank_idx + 1
            reasoning = generate_reasoning(sc["candidate"], rank, sc["scores"])
            results.append({
                "rank": rank,
                "candidate_id": sc["candidate"]["candidate_id"],
                "score": sc["final_score"],
                "scores": sc["scores"],
                "reasoning": reasoning,
                "candidate": sc["candidate"],
            })

        progress_bar.progress(100, "✅ Pipeline complete!")
        time.sleep(0.5)
        progress_bar.empty()

        # Store in session
        st.session_state["results"] = results
        st.session_state["stats"] = {
            "total": total,
            "filtered": n_filtered,
            "eliminated": total - n_filtered,
            "honeypots": honeypot_count,
        }

    results = st.session_state["results"]
    stats = st.session_state["stats"]

    # ── Summary Metrics ──────────────────────────────────────────────────
    st.markdown("## 📈 Pipeline Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    render_metric_card(f"{stats['total']:,}", "Total Candidates", c1)
    render_metric_card(f"{stats['filtered']:,}", "Passed Filters", c2)
    render_metric_card(f"{stats['eliminated']:,}", "Eliminated", c3)
    render_metric_card(str(stats['honeypots']), "Honeypots Found", c4)
    render_metric_card(f"{results[0]['score']:.3f}", "Top Score", c5)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏅 Rankings", "📊 Score Analysis", "🕵️ Honeypot Inspector", "📋 Export"
    ])

    # ── Tab 1: Rankings ──────────────────────────────────────────────────
    with tab1:
        st.markdown("### Top 100 Ranked Candidates")

        # Quick stats
        scores_list = [r["score"] for r in results]
        st.markdown(f"""
        **Score range**: {max(scores_list):.4f} → {min(scores_list):.4f} |
        **Median**: {sorted(scores_list)[50]:.4f} |
        **Spread**: {max(scores_list) - min(scores_list):.4f}
        """)

        # Rankings table
        df_display = pd.DataFrame([
            {
                "Rank": r["rank"],
                "ID": r["candidate_id"],
                "Score": f"{r['score']:.4f}",
                "Title": (r["candidate"].get("profile", {}).get("current_title", "N/A") or "N/A"),
                "Company": (r["candidate"].get("profile", {}).get("current_company", "N/A") or "N/A"),
                "Years": r["candidate"].get("profile", {}).get("years_of_experience", 0) or 0,
                "Location": (r["candidate"].get("profile", {}).get("location", "N/A") or "N/A"),
                "Reasoning": r["reasoning"],
            }
            for r in results
        ])
        st.dataframe(df_display, use_container_width=True, height=500)

        # Candidate deep-dive
        st.markdown("### 🔍 Candidate Deep Dive")
        selected_rank = st.selectbox(
            "Select a candidate to inspect",
            range(1, 101),
            format_func=lambda x: f"Rank #{x}: {results[x-1]['candidate_id']} — {results[x-1]['candidate'].get('profile', {}).get('current_title', 'N/A')}"
        )

        r = results[selected_rank - 1]
        c = r["candidate"]
        s = r["scores"]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"#### {c.get('profile', {}).get('current_title', 'N/A')}")
            st.markdown(f"**Company**: {c.get('profile', {}).get('current_company', 'N/A')}")
            st.markdown(f"**Experience**: {c.get('profile', {}).get('years_of_experience', 0):.1f} years")
            st.markdown(f"**Location**: {c.get('profile', {}).get('location', 'N/A')}")
            st.markdown(f"**Final Score**: `{r['score']:.4f}`")
            st.info(f"💬 {r['reasoning']}")

            # Skills
            skills = c.get("skills", [])
            if skills:
                skill_names = [s.get("name", "?") for s in skills[:10]]
                st.markdown("**Top Skills**: " + " · ".join(f"`{s}`" for s in skill_names))

        with col2:
            st.markdown("#### Dimension Breakdown")
            render_score_bar("Career Coherence", s["career_coherence"], 25, custom_weights["career_coherence"])
            render_score_bar("Skill Authenticity", s["skill_authenticity"], 25, custom_weights["skill_authenticity"])
            render_score_bar("JD Alignment", s["jd_semantic_align"], 20, custom_weights["jd_semantic_align"])
            render_score_bar("Behavioral Availability", s["behavioral_avail"], 15, custom_weights["behavioral_avail"])
            render_score_bar("Location Fit", s["location_fit"], 10, custom_weights["location_fit"])
            render_score_bar("Education & Credibility", s["education_cred"], 5, custom_weights["education_cred"])

    # ── Tab 2: Score Analysis ────────────────────────────────────────────
    with tab2:
        st.markdown("### Score Distribution")

        # Histogram of final scores
        scores_df = pd.DataFrame({"Final Score": [r["score"] for r in results]})
        st.bar_chart(scores_df, y="Final Score")

        # Dimension comparison
        st.markdown("### Dimension Averages (Top 100)")
        dim_avgs = {}
        for dim in WEIGHTS:
            vals = [r["scores"][dim] for r in results]
            dim_avgs[dim] = sum(vals) / len(vals)

        dim_df = pd.DataFrame([
            {
                "Dimension": dim.replace("_", " ").title(),
                "Avg Score": avg,
                "Max Possible": MAX_POINTS[dim],
                "Fill %": avg / MAX_POINTS[dim] * 100,
            }
            for dim, avg in dim_avgs.items()
        ])
        st.dataframe(dim_df, use_container_width=True)

        # Weight pie chart
        st.markdown("### Current Weight Distribution")
        weight_df = pd.DataFrame([
            {"Dimension": k.replace("_", " ").title(), "Weight": v}
            for k, v in custom_weights.items()
        ])
        st.bar_chart(weight_df.set_index("Dimension"))

    # ── Tab 3: Honeypot Inspector ────────────────────────────────────────
    with tab3:
        st.markdown("### 🕵️ Honeypot Detection Results")
        st.markdown(f"""
        Detected **{stats['honeypots']}** honeypots out of {stats['filtered']:,} scored candidates
        ({stats['honeypots']/max(stats['filtered'],1)*100:.2f}%).

        All honeypots have been excluded from the top 100 rankings.
        Honeypot rate in top 100: **0%** (threshold: <10%).
        """)

        st.markdown("### How Detection Works")
        st.markdown("""
        The system runs **6 independent heuristic checks** on each candidate:

        | # | Check | What It Catches |
        |---|-------|-----------------|
        | 1 | Experience-Date Mismatch | Claims 10yr exp but career history totals 4yr |
        | 2 | Skill Inflation | 'Expert' in 5+ skills with <3 months each |
        | 3 | Title-Description Mismatch | Marketing Manager with ML pipeline descriptions |
        | 4 | Impossible Timeline | Role says 60mo but dates span 12mo |
        | 5 | Keyword Stuffing | 20+ AI skills but no career mentions |
        | 6 | Abnormal Signal Combo | Perfect scores on a suspicious profile |

        A candidate is flagged as a honeypot if **2 or more checks trigger**.
        """)

    # ── Tab 4: Export ────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 📋 Download Results")

        # Generate CSV
        csv_data = "candidate_id,rank,score,reasoning\n"
        for r in results:
            reasoning_escaped = r["reasoning"].replace('"', '""')
            csv_data += f'{r["candidate_id"]},{r["rank"]},{r["score"]:.6f},"{reasoning_escaped}"\n'

        st.download_button(
            label="📥 Download submission.csv",
            data=csv_data,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("### Preview")
        st.code(csv_data[:2000] + "\n...", language="csv")

else:
    # Landing page
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h2 style="color: #e2e8f0;">Welcome to the Redrob AI Ranker</h2>
        <p style="color: #a0aec0; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            This tool demonstrates a 5-stage intelligent ranking pipeline that
            evaluates candidates across 6 dimensions — career coherence, skill
            authenticity, JD alignment, behavioral signals, location fit, and
            education credibility.
        </p>
        <br>
        <p style="color: #a0aec0;">
            👈 Click <strong>"Run Pipeline"</strong> in the sidebar to start
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture overview
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔍 Stage 1-3
        **Filter → Score → Rank**

        Hard filters eliminate non-starters, then each
        candidate is scored across 6 weighted dimensions
        using TF-IDF semantic matching and signal fusion.
        """)

    with col2:
        st.markdown("""
        ### 🕵️ Stage 4
        **Honeypot Detection**

        6-check anomaly system catches impossible profiles
        (experience-date mismatches, skill inflation,
        title-description contradictions).
        """)

    with col3:
        st.markdown("""
        ### 💬 Stage 5
        **Reasoning Generation**

        Each ranked candidate gets a specific, honest,
        varied explanation citing real facts from their
        profile. No hallucination, no templates.
        """)
