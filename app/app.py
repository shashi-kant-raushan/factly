"""
Fake News / Misinformation Detection Classifier - Streamlit Web Application
Assigned Mentor: Sudipta Ghosh
Type: ML / NLP — Binary Text Classification

Tech Stack:
pandas, NumPy, scikit-learn (TF-IDF, Logistic Regression, Random Forest), Hugging Face DistilBERT, SHAP

Implementation Approach:
TF-IDF + Logistic Regression baseline -> Random Forest -> fine-tuned DistilBERT -> interpretability via feature importance/attention/SHAP.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    TFIDF_VECTORIZER_PATH,
    LOGISTIC_MODEL_PATH,
    RANDOM_FOREST_PATH,
    DISTILBERT_DIR,
    EVALUATION_METRICS_PATH,
    FIGURES_DIR,
    METADATA_JSON
)
from src.explainability import (
    explain_prediction,
    get_global_logistic_features,
    get_global_rf_features
)


# Set page configuration
st.set_page_config(
    page_title="Factly",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphic aesthetics & styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Main Title Styling */
    .app-header {
        position: relative;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(27, 37, 75, 0.96) 45%, rgba(58, 28, 92, 0.95) 100%);
        padding: 2.2rem 2.5rem 2rem 2.5rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 18px 30px -16px rgba(15, 23, 42, 0.8), inset 0 1px 0 rgba(255,255,255,0.08);
        border: 1px solid rgba(148, 163, 184, 0.18);
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .app-header::before {
        content: "";
        position: absolute;
        width: 460px;
        height: 460px;
        top: -170px;
        right: -80px;
        background: radial-gradient(circle, rgba(96, 165, 250, 0.25), rgba(167, 139, 250, 0.1) 40%, transparent 72%);
        pointer-events: none;
    }

    .brand-wrap {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        text-align: center;
        flex-wrap: wrap;
    }

    .brand-badge {
        width: 62px;
        height: 62px;
        border-radius: 18px;
        background: linear-gradient(135deg, #7dd3fc 0%, #c4b5fd 50%, #f9a8d4 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 24px rgba(124, 58, 237, 0.4);
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 800;
    }

    .app-title {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.08;
        background: linear-gradient(90deg, #7dd3fc 0%, #c4b5fd 45%, #f9a8d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.05em;
        position: relative;
        z-index: 1;
    }

    .app-subtitle {
        font-size: 1.05rem;
        color: #E2E8F0;
        margin-top: 1rem;
        line-height: 1.7;
        margin-bottom: 0;
        max-width: 1180px;
        position: relative;
        z-index: 1;
        text-align: center;
    }

    .stTabs [role="tablist"] {
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    .stTabs [role="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.7rem 1.1rem;
        font-weight: 600;
        color: #475569;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.12), rgba(167, 139, 250, 0.12));
        border: 1px solid rgba(96, 165, 250, 0.2);
        color: #0F172A;
    }

    /* Result Card Styling */
    .result-box-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 1.8rem;
        margin-top: 1.2rem;
        box-shadow: 0 8px 20px -4px rgba(16, 185, 129, 0.15);
    }
    
    .result-box-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 1.8rem;
        margin-top: 1.2rem;
        box-shadow: 0 8px 20px -4px rgba(239, 68, 68, 0.15);
    }

    .badge-real {
        display: inline-block;
        background-color: #10B981;
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-fake {
        display: inline-block;
        background-color: #EF4444;
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Token Attribution Chips */
    .token-chip-fake {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        color: #DC2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.3rem 0.7rem;
        margin: 0.25rem;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .token-chip-real {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.3rem 0.7rem;
        margin: 0.25rem;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* Metric Cards */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1E293B;
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        margin-top: 0.25rem;
    }

    /* Mentor Specs Table */
    .specs-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
        background: #FFFFFF;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .specs-table th {
        background: #0F172A;
        color: #FFFFFF;
        padding: 0.9rem 1.2rem;
        text-align: left;
        font-size: 0.95rem;
    }
    .specs-table td {
        padding: 1rem 1.2rem;
        border-top: 1px solid #E2E8F0;
        color: #334155;
        font-size: 0.95rem;
        vertical-align: top;
    }

    /* Disclaimer Alert */
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1.5rem;
        color: #92400E;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Preset news article examples
PRESET_EXAMPLES = {
    "— Select a Sample Article —": "",
    "Real News: NASA James Webb Telescope Discovery": (
        "NASA's James Webb Space Telescope has captured unprecedented detailed spectroscopic observations "
        "of a distant exoplanet's atmosphere, confirming the presence of water vapor, sulfur dioxide, and complex "
        "carbon compounds. According to a peer-reviewed study published in the journal Nature, international research teams "
        "analyzed data transmitted from the observatory located 1 million miles from Earth. Officials confirmed the findings "
        "align with theoretical atmospheric models."
    ),
    "Real News: Central Bank Interest Rate Policy Announcement": (
        "The Federal Reserve announced on Wednesday that benchmark interest rates would remain unchanged following "
        "a two-day monetary policy committee meeting. In an official press conference, the committee chairman cited "
        "moderating inflation metrics and steady employment growth across major economic sectors. Reuters reported that "
        "financial markets responded positively to the announcement, with treasury yields stabilizing across short-term maturities."
    ),
    "Fake News: Secret Miracle Cure Hidden by Doctors": (
        "SHOCKING BREAKING DISCOVERY: Doctors are furious after a whistleblower leaked a secret miracle cure that Big Pharma "
        "has been hiding from the public for decades! This bizarre ancient ritual cures 100% of illnesses overnight with zero pills. "
        "Mainstream media is actively censoring this video before it gets deleted! Click immediately to uncover the terrifying truth "
        "the government doesn't want you to see!"
    ),
    "Fake News: Galactic Alien Mind Control Intercepted": (
        "UNBELIEVABLE: Secret military insiders reveal that an encrypted extraterrestrial broadcast was intercepted last night "
        "confirming that reptilian aliens have secretly taken over world governments! The leaked memo proves that global leaders "
        "are actually clones operating under high-frequency mind control. Share this emergency warning with everyone before "
        "agents shut down the global internet grid!"
    )
}


def load_metrics_data() -> dict:
    """Loads saved model benchmark metrics if available."""
    if EVALUATION_METRICS_PATH.exists():
        with open(EVALUATION_METRICS_PATH, "r") as f:
            return json.load(f)
    return {}


def load_eda_metadata() -> dict:
    """Loads dataset EDA metadata if available."""
    if METADATA_JSON.exists():
        with open(METADATA_JSON, "r") as f:
            return json.load(f)
    return {}


def render_header():
    """Renders the cleaned project header."""
    st.markdown("""
    <div class="app-header">
        <div class="brand-wrap">
            <div class="brand-badge">F</div>
            <div>
                <h1 class="app-title">Factly</h1>
            </div>
        </div>
        <p class="app-subtitle">
            Detect misleading or false narratives with AI-powered news verification.
            Factly combines classical and transformer-based classifiers to identify likely misinformation and highlight the language signals behind each prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)


def check_models_availability():
    """Checks which models are currently trained and available."""
    return {
        "TF-IDF + Logistic Regression": TFIDF_VECTORIZER_PATH.exists() and LOGISTIC_MODEL_PATH.exists(),
        "TF-IDF + Random Forest": TFIDF_VECTORIZER_PATH.exists() and RANDOM_FOREST_PATH.exists(),
        "Fine-tuned DistilBERT": DISTILBERT_DIR.exists()
    }


def main():
    render_header()
    available_models = check_models_availability()

    # Sidebar Navigation & Settings
    st.sidebar.title("⚙️ Model Selector")
    
    # 3 core models + Side-by-side comparison (exactly as specified in the slide)
    model_options = [
        "Fine-tuned DistilBERT (Transformer)",
        "TF-IDF + Logistic Regression (Linear Baseline)",
        "TF-IDF + Random Forest (Ensemble Tree)",
        "⚡ Compare All 3 Models (Side-by-Side)"
    ]

    selected_model = st.sidebar.selectbox(
        "Choose Classification Model:",
        options=model_options,
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Model Status")
    for name, is_avail in available_models.items():
        status_icon = "🟢 Ready" if is_avail else "🔴 Missing"
        st.sidebar.write(f"**{name}:** {status_icon}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Implementation Approach:**
    `TF-IDF + Logistic Regression baseline` ➔ `Random Forest` ➔ `Fine-tuned DistilBERT` ➔ `Interpretability via Feature Importance / Attention / SHAP`
    """)

    # Main Tabs (Aligned strictly with the photo requirements)
    tab_analyzer, tab_benchmarks, tab_explain, tab_specs = st.tabs([
        "🔍 News Classifier",
        "📊 Model Benchmarks (Accuracy, F1, Precision, Recall)",
        "🧠 Interpretability (SHAP, Feature Importance & Attention)",
        "📋 Mentor Specifications & Approach"
    ])

    # -------------------------------------------------------------
    # TAB 1: NEWS CLASSIFIER
    # -------------------------------------------------------------
    with tab_analyzer:
        st.markdown("### 📰 Input Article for Credibility Classification")
        st.write("Enter an article headline or body text below, or select a preset example to classify.")

        # Preset Dropdown
        selected_preset = st.selectbox("Quick Load Preset Sample:", list(PRESET_EXAMPLES.keys()))
        default_text = PRESET_EXAMPLES[selected_preset]

        user_input = st.text_area(
            "Article Text / Headline:",
            value=default_text,
            height=170,
            placeholder="Paste raw news text or headline here for credibility classification..."
        )

        col_btn, _ = st.columns([1, 5])
        with col_btn:
            analyze_clicked = st.button("🚀 Classify News", type="primary", use_container_width=True)

        if analyze_clicked:
            if not user_input.strip():
                st.error("❌ Please provide article text before classifying.")
            elif not any(available_models.values()):
                st.error("❌ Model artifacts not found. Please verify `models/` directory.")
            else:
                with st.spinner("Classifying linguistic patterns and computing token attributions..."):
                    start_time = time.perf_counter()
                    result = explain_prediction(user_input, model_choice=selected_model)
                    latency = (time.perf_counter() - start_time) * 1000

                if "Side-by-Side" in selected_model or "Compare All 3" in selected_model:
                    st.markdown("### ⚡ Side-by-Side Comparison of All 3 Models")
                    cols = st.columns(3)
                    model_names = ["Logistic Regression", "Random Forest", "DistilBERT"]

                    for idx, m_name in enumerate(model_names):
                        res = result.get(m_name, {})
                        if not res or "error" in res:
                            cols[idx].error(f"Error in {m_name}: {res.get('error', 'Unknown')}")
                            continue

                        pred = res.get("prediction", "UNKNOWN")
                        conf = res.get("confidence", 0.0)
                        probs = res.get("probabilities", {"REAL": 50.0, "FAKE": 50.0})
                        is_real = (pred == "REAL")
                        box_class = "result-box-real" if is_real else "result-box-fake"
                        badge_class = "badge-real" if is_real else "badge-fake"
                        badge_icon = "✅" if is_real else "🚨"

                        with cols[idx]:
                            st.markdown(f"""
                            <div class="{box_class}">
                                <h4 style="margin: 0 0 0.5rem 0; color: #1E293B;">{res.get('model_type', m_name)}</h4>
                                <span class="{badge_class}">{badge_icon} {pred}</span>
                                <h3 style="margin: 0.8rem 0 0 0; color: {'#065F46' if is_real else '#991B1B'};">
                                    {conf:.1f}% Confidence
                                </h3>
                                <div style="margin-top: 0.8rem; font-size: 0.85rem;">
                                    <strong>Probabilities:</strong><br>
                                    REAL: {probs['REAL']:.1f}% | FAKE: {probs['FAKE']:.1f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown("**Top Salient Tokens:**")
                            tokens = res.get("top_contributing_words", [])[:4]
                            chips_html = ""
                            for tok_info in tokens:
                                w = tok_info.get("word") or tok_info.get("clean_token")
                                score = tok_info.get("score", 0.0)
                                favors = tok_info.get("favors", "FAKE")
                                chip_class = "token-chip-fake" if favors == "FAKE" else "token-chip-real"
                                chips_html += f'<span class="{chip_class}">{w} ({score:+.2f})</span> '
                            st.markdown(chips_html if chips_html else "*None*", unsafe_allow_html=True)
                            st.caption(f"Method: *{res.get('method', 'N/A')}*")

                elif "error" in result:
                    st.error(f"❌ Error during inference: {result['error']}")
                else:
                    pred = result.get("prediction", "UNKNOWN")
                    conf = result.get("confidence", 0.0)
                    probs = result.get("probabilities", {"REAL": 50.0, "FAKE": 50.0})
                    is_real = (pred == "REAL")
                    box_class = "result-box-real" if is_real else "result-box-fake"
                    badge_class = "badge-real" if is_real else "badge-fake"
                    badge_icon = "✅" if is_real else "🚨"

                    # Result Card
                    st.markdown(f"""
                    <div class="{box_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="{badge_class}">{badge_icon} PREDICTION: {pred}</span>
                                <h2 style="margin: 0.8rem 0 0 0; color: {'#065F46' if is_real else '#991B1B'};">
                                    Confidence: {conf:.1f}%
                                </h2>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 0.9rem; color: #64748B; font-weight: 600;">Model Used</span><br>
                                <span style="font-size: 1.1rem; font-weight: 700; color: #1E293B;">{result.get('model_type', selected_model)}</span><br>
                                <span style="font-size: 0.85rem; color: #94A3B8;">Latency: {latency:.1f} ms</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability Distribution Bars
                    st.markdown("#### 📈 Probability Distribution")
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.write(f"**REAL News Probability:** {probs['REAL']:.1f}%")
                        st.progress(float(probs['REAL']) / 100.0)
                    with col_p2:
                        st.write(f"**FAKE News Probability:** {probs['FAKE']:.1f}%")
                        st.progress(float(probs['FAKE']) / 100.0)

                    # Words / phrases most predictive via XAI
                    st.markdown("---")
                    st.markdown("#### 🔍 Words / Phrases Most Predictive of Credibility")
                    st.write("Identified via feature importance / SHAP / attention as specified in project objective:")

                    tokens = result.get("top_contributing_words", [])
                    if tokens:
                        token_chips_html = ""
                        for tok_info in tokens:
                            w = tok_info.get("word") or tok_info.get("clean_token")
                            score = tok_info.get("score", 0.0)
                            favors = tok_info.get("favors", "FAKE")
                            chip_class = "token-chip-fake" if favors == "FAKE" else "token-chip-real"
                            token_chips_html += f'<span class="{chip_class}">{w} ({score:+.3f})</span>'
                        st.markdown(token_chips_html, unsafe_allow_html=True)
                    else:
                        st.info("No salient vocabulary tokens exceeded threshold for this input.")

                    method_str = result.get("method", "Feature Attribution")
                    st.caption(f"Explainability Method: *{method_str}*")

    # -------------------------------------------------------------
    # TAB 2: MODEL BENCHMARKS
    # -------------------------------------------------------------
    with tab_benchmarks:
        st.markdown("### 📊 Model Performance Benchmarks on Unseen Test Data")
        st.write("Reporting **Accuracy, F1, Precision, and Recall** as outlined in the project objective:")

        metrics_data = load_metrics_data()

        if metrics_data:
            table_rows = []
            for k, v in metrics_data.items():
                if "Ensemble" in k or "Ensemble" in v.get("model_name", ""):
                    continue  # Keep strictly the 3 models from the slide
                table_rows.append({
                    "Model": v.get("model_name", k).replace("🔥 ", ""),
                    "Accuracy (%)": f"{v.get('accuracy', 0)*100:.2f}%",
                    "F1-Score": f"{v.get('f1', 0):.4f}",
                    "Precision": f"{v.get('precision', 0):.4f}",
                    "Recall": f"{v.get('recall', 0):.4f}",
                    "ROC-AUC": f"{v.get('roc_auc', 0):.4f}",
                    "Latency (ms)": f"{v.get('latency_ms', 0)} ms",
                    "Model Size": f"{v.get('model_size_mb', 0)} MB"
                })
            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### 📉 Comparative Benchmark Charts")
            col_fig1, col_fig2 = st.columns(2)
            perf_chart_path = FIGURES_DIR / "model_performance_comparison.png"
            roc_chart_path = FIGURES_DIR / "roc_curves.png"
            cm_chart_path = FIGURES_DIR / "confusion_matrices.png"

            with col_fig1:
                if perf_chart_path.exists():
                    st.image(str(perf_chart_path), caption="Accuracy, Precision, Recall, F1 & ROC-AUC Comparison", use_container_width=True)
            with col_fig2:
                if roc_chart_path.exists():
                    st.image(str(roc_chart_path), caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)

            if cm_chart_path.exists():
                st.markdown("#### 🎯 Side-by-Side Confusion Matrices")
                st.image(str(cm_chart_path), caption="Test Set Confusion Matrices across Baselines & Transformer", use_container_width=True)
        else:
            st.info("Metrics not yet generated. Run `python src/evaluate.py` to evaluate.")

    # -------------------------------------------------------------
    # TAB 3: INTERPRETABILITY (SHAP, FEATURE IMPORTANCE, ATTENTION)
    # -------------------------------------------------------------
    with tab_explain:
        st.markdown("### 🧠 Interpretability: Identify Words & Phrases Predictive of 'Fake'")
        st.write("Comparing the 3 interpretability techniques specified in the objective:")

        st.markdown("#### 1. SHAP & Logistic Regression Feature Coefficients")
        col_g1, col_g2 = st.columns(2)
        global_features = get_global_logistic_features(top_n=12)
        fake_inds = global_features.get("fake_indicators", [])
        real_inds = global_features.get("real_indicators", [])

        with col_g1:
            st.markdown("🚨 **Top Words Predictive of 'FAKE' (Positive SHAP/Coefs)**")
            if fake_inds:
                df_fake = pd.DataFrame(fake_inds).rename(columns={"word": "Predictive Word/Phrase", "weight": "SHAP Coefficient (+)"})
                st.dataframe(df_fake, use_container_width=True, hide_index=True)

        with col_g2:
            st.markdown("✅ **Top Words Predictive of 'REAL' (Negative SHAP/Coefs)**")
            if real_inds:
                df_real = pd.DataFrame(real_inds).rename(columns={"word": "Predictive Word/Phrase", "weight": "SHAP Coefficient (-)"})
                st.dataframe(df_real, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 2. Random Forest Tree Feature Importance (MDI)")
        rf_features = get_global_rf_features(top_n=12)
        if rf_features:
            df_rf = pd.DataFrame(rf_features).rename(columns={"word": "N-Gram Feature", "importance": "MDI Feature Importance"})
            st.dataframe(df_rf, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 3. DistilBERT Token Embedding Gradient Saliency & Attention")
        st.info("DistilBERT computes contextual gradient saliency $( \\nabla_x \\mathcal{L} \\cdot x )$ for each token during live inference in Tab 1.")

    # -------------------------------------------------------------
    # TAB 4: MENTOR SPECS & SLIDE DETAILS
    # -------------------------------------------------------------
    with tab_specs:
        st.markdown("### 📋 Mentor Project Specification Slide")
        st.markdown("""
        <table class="specs-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Use Case</th>
                    <th>Use case Description / Problem Statement</th>
                    <th>Objective</th>
                    <th>Type</th>
                    <th>Mentor</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>3</strong></td>
                    <td><strong>Fake News / Misinformation Detection Classifier</strong></td>
                    <td>
                        Misinformation spreads quickly, and manually fact-checking every article doesn't scale.<br>
                        This project frames detection as binary text classification, comparing a linear baseline, an ensemble tree model, and a fine-tuned Transformer to determine which best captures the linguistic patterns that separate misinformation from legitimate reporting.
                    </td>
                    <td>
                        Vectorize text with TF-IDF; train Logistic Regression and Random Forest baselines; fine-tune DistilBERT; report Accuracy, F1, Precision, Recall; identify words/phrases most predictive of "fake" via feature importance/SHAP/attention.
                    </td>
                    <td><strong>ML / NLP — Binary Text Classification</strong></td>
                    <td><strong>Sudipta Ghosh</strong></td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### 🛠️ TECH STACK & IMPLEMENTATION APPROACH
        - **Tech Stack:** `pandas`, `NumPy`, `scikit-learn (TF-IDF, Logistic Regression, Random Forest)`, `Hugging Face DistilBERT`, `SHAP`.
        - **Approach:** `TF-IDF + Logistic Regression baseline` ➔ `Random Forest` ➔ `fine-tuned DistilBERT` ➔ `interpretability via feature importance/attention/SHAP`.
        """)


if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(stcli.main())

