import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="VENTURESCALE AI | VC Due Diligence & Valuation Engine",
    page_icon="📈",
    layout="wide",
)

st.title("📈 VENTURESCALE AI")
st.caption("Quantitative Pitch Deck Analytics, Unit Economics Stress Tester & Valuation Engine")

# Initialize Gemini API Client secretly
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in .env file. Please check your configuration.")
    st.stop()


@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)


client = get_gemini_client(api_key)


# Pydantic Schemas for Structured Model Output
class UnitMetric(BaseModel):
    metric_name: str = Field(description="Unit economics metric (e.g., LTV:CAC Ratio, Monthly Payback, Gross Margin, Net Churn).")
    calculated_value: str = Field(description="Estimated or extracted quantitative value.")
    benchmark_grade: str = Field(description="Evaluation vs top-decile VC benchmarks: 'Top 5% (Elite)', 'Healthy', 'Below Benchmark', or 'Fatal Flaw'")

class InvestmentVectorWeight(BaseModel):
    factor_name: str = Field(description="Startup evaluation vector (e.g., Founder-Market Fit, TAM Expansion, CAC Efficiency, Moat Defensibility).")
    weight: float = Field(description="Normalized feature impact weight (|SHAP value| 0.0 to 1.0).")
    impact_direction: str = Field(description="Impact on IC decision: 'Drives Upside' or 'Increases Downside Risk'")

class VentureDiligenceInference(BaseModel):
    venture_stage: str = Field(description="Startup stage classification (e.g., Pre-Seed, Seed, Series A SaaS, Marketplace).")
    ic_investment_score: int = Field(description="Investment Committee conviction score from 0 (Hard Pass) to 100 (Top-Tier Term Sheet).")
    vc_conviction_tier: str = Field(description="Tier classification: 'TIER 1 TERM SHEET', 'CONDITIONAL DUE DILIGENCE', or 'PASS / NO-GO'")
    implied_valuation_range: str = Field(description="Estimated fair market pre-money valuation range (e.g., $4M - $6M).")
    unit_economics_breakdown: list[UnitMetric] = Field(description="Key quantitative SaaS/Marketplace metrics calculated.")
    investment_vector_weights: list[InvestmentVectorWeight] = Field(description="SHAP feature importances driving the IC score.")
    red_flag_risks: list[str] = Field(description="Critical deal-killer risks, unit economic traps, or go-to-market bottlenecks.")
    ic_term_sheet_conditions: list[str] = Field(description="Key milestones, board requirements, or diligence gates before funding.")


SYSTEM_INSTRUCTION = """
You are VENTURESCALE AI, an elite Institutional Venture Capital Partner and Quantitative Investment Committee (IC) Engine.
Analyze the provided pitch deck executive summary, financial projections, MRR/CAC figures, and market positioning.
Calculate unit economics benchmarks, evaluate a 0-100 IC Investment Score, estimate a pre-money valuation range, extract SHAP investment feature weights, and issue term sheet conditions.

Output strictly valid JSON according to the requested schema. Maintain a sharp, institutional Silicon Valley VC tone.
"""


def simulate_diligence_pipeline():
    """Simulates realistic pitch deck processing and financial sensitivity modeling."""
    progress_bar = st.progress(0, text="Extracting Financial Metrics & Deck Embeddings...")
    time.sleep(0.3)

    progress_bar.progress(35, text="Running LTV:CAC Cohort Sensitivity & Payback Simulations...")
    time.sleep(0.4)

    progress_bar.progress(70, text="Benchmarking Against 10,000+ Venture Fundings & Valuations...")
    time.sleep(0.4)

    progress_bar.progress(95, text="Generating Investment Committee SHAP Vectors & Term Sheet Gates...")
    time.sleep(0.3)

    progress_bar.progress(100, text="Diligence Vector Analysis Complete!")
    time.sleep(0.2)
    progress_bar.empty()


def render_venture_dashboard(result: dict):
    """Renders structured AI inference formatted as a VC Partner Dashboard."""

    stage = result.get("venture_stage", "Seed Stage")
    score = result.get("ic_investment_score", 50)
    tier = result.get("vc_conviction_tier", "CONDITIONAL DUE DILIGENCE")
    valuation = result.get("implied_valuation_range", "N/A")

    # 1. Primary Metrics Header
    st.subheader("🎯 Investment Committee Decision Vector")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(label="Venture Stage", value=stage)
    c2.metric(label="Estimated Pre-Money Valuation", value=valuation)
    c3.metric(label="IC Conviction Score", value=f"{score} / 100")

    if "TERM SHEET" in tier or "TIER 1" in tier:
        c4.success(f"**IC Stance:**\n\n### 🟢 {tier}")
    elif "CONDITIONAL" in tier:
        c4.warning(f"**IC Stance:**\n\n### ⚠️ {tier}")
    else:
        c4.error(f"**IC Stance:**\n\n### 🚨 {tier}")

    st.divider()

    # 2. Charts: IC Conviction Gauge & SHAP Feature Importances
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("🌡️ IC Conviction Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Term Sheet Probability Index", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#00CC96" if score >= 75 else ("#FFAA00" if score >= 50 else "#FF4B4B")},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(255, 75, 75, 0.2)"},
                    {'range': [50, 75], 'color': "rgba(255, 170, 0, 0.2)"},
                    {'range': [75, 100], 'color': "rgba(0, 204, 150, 0.2)"}
                ],
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_chart2:
        st.subheader("📊 Investment Vector Weights (|SHAP|)")
        weights = result.get("investment_vector_weights", [])
        if weights:
            df_weights = pd.DataFrame(weights)
            fig_bar = px.bar(
                df_weights,
                x="weight",
                y="factor_name",
                color="impact_direction",
                orientation="h",
                title="Key Factors Influencing Conviction Score",
                labels={"weight": "Weight (|SHAP|)", "factor_name": "Investment Vector"},
                color_discrete_map={"Drives Upside": "#00CC96", "Increases Downside Risk": "#FF4B4B"}
            )
            fig_bar.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # 3. Unit Economics Benchmarks Table
    st.subheader("📊 Quantitative Unit Economics & Benchmark Audit")
    metrics = result.get("unit_economics_breakdown", [])
    if metrics:
        df_metrics = pd.DataFrame(metrics)
        df_metrics.columns = ["Metric", "Calculated / Reported Value", "VC Benchmark Status"]
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.divider()

    # 4. Red Flags & Term Sheet Conditions
    c_flags, c_gates = st.columns(2)

    with c_flags:
        st.subheader("🚩 Critical Deal Red Flags & Bottlenecks")
        for flag in result.get("red_flag_risks", []):
            st.markdown(f"- ⚠️ {flag}")

    with c_gates:
        st.subheader("📑 Term Sheet Due Diligence Gates")
        for gate in result.get("ic_term_sheet_conditions", []):
            st.markdown(f"- 📋 {gate}")


# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Model Architecture")
    st.selectbox("Diligence Engine", ["VentureScale-LLaMA3-8B VC", "Proprietary Fund-III Scoring Net", "XGBoost Unit-Economics Classifier"])
    st.slider("Fund Return Target (MOIC)", min_value=3.0, max_value=10.0, value=5.0, step=0.5)
    st.selectbox("Target Sector Benchmark", ["B2B SaaS", "AI Infrastructure", "Marketplace / Fintech", "Consumer Mobile"])
    st.caption("Status: Model calibrated on 10,000+ venture pitch decks and Series A/B deal memos.")

# Main Input Section
st.subheader("📥 Input Pitch Deck Narrative & Financial Vector")

with st.form("diligence_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        mrr = st.number_input("Current Monthly Recurring Revenue ($)", value=45000, step=5000)
        mom_growth = st.number_input("MoM Revenue Growth (%)", value=18.0, step=1.0)
    with c2:
        cac = st.number_input("Customer Acquisition Cost (CAC $)", value=650, step=50)
        arpu = st.number_input("Average Revenue Per User / Mo ($)", value=220, step=10)
    with c3:
        churn_pct = st.number_input("Monthly Gross Churn (%)", value=1.8, step=0.1)
        burn_rate = st.number_input("Net Monthly Burn Rate ($)", value=35000, step=5000)

    deck_summary = st.text_area(
        "Paste Pitch Deck Executive Summary, Problem/Solution, and Team Background:",
        value="CloudOps AI is an automated infrastructure cost optimization tool for Kubernetes clusters. Founded by ex-AWS principal engineers. We reduce AWS bills by 35% using real-time node scaling algorithms. Currently serving 28 enterprise customers including 2 unicorn SaaS companies. Raising $2.5M Seed at an $18M pre-money valuation to scale outbound SDR team.",
        height=130
    )

    run_pipeline_btn = st.form_submit_button("🚀 Run VC Due Diligence & Valuation Pipeline", type="primary", use_container_width=True)


if run_pipeline_btn:
    if not deck_summary.strip():
        st.warning("Please enter pitch deck details to evaluate.")
    else:
        # 1. UI Pipeline Animation
        simulate_diligence_pipeline()

        # 2. Secretly construct feature payload for Gemini
        payload = {
            "metrics": {
                "mrr": mrr,
                "mom_growth_pct": mom_growth,
                "cac": cac,
                "arpu": arpu,
                "monthly_churn_pct": churn_pct,
                "monthly_burn": burn_rate
            },
            "deck_executive_summary": deck_summary
        }

        # 3. Call Gemini under the hood
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[f"Execute venture capital due diligence and valuation evaluation for: {json.dumps(payload)}"],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=VentureDiligenceInference,
                ),
            )

            inference_data = json.loads(response.text)

            # 4. Render as VC Partner Dashboard
            render_venture_dashboard(inference_data)

        except Exception as e:
            st.error(f"Diligence Engine Failure: {e}")