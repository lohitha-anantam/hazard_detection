import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from agent_workflow import run_incident_workflow
from config import REPORTS_DIR

st.set_page_config(page_title="Road Hazard Response Copilot", page_icon="🛣️", layout="wide")
st.title("🛣️ Road Hazard Detection & Response Copilot")
st.caption("Computer vision · ML triage · RAG · LLM reporting · LangGraph orchestration")

st.warning(
    "Decision-support prototype only. AI observations and risk scores can be wrong. "
    "A human must verify results. This app does not contact authorities or dispatch responders."
)

with st.sidebar:
    st.header("Incident details")
    hazard_type = st.selectbox(
        "Reported hazard type",
        ["Unknown", "Pothole / road damage", "Debris / obstruction", "Damaged manhole", "Flooding", "Traffic incident", "Other"],
    )
    location = st.text_input("Location (as reported)", placeholder="Road name / landmark / area")
    user_notes = st.text_area(
        "Additional observations",
        placeholder="What did you observe? Is the road blocked? Any immediate danger?",
        height=130,
    )
    st.caption("Do not enter sensitive personal details or approach a dangerous road to collect evidence.")

uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)

if st.button("Analyze incident", type="primary", use_container_width=True):
    if uploaded is None:
        st.error("Upload a road image first.")
        st.stop()

    suffix = Path(uploaded.name).suffix.lower() or ".jpg"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded.getvalue())
            temp_path = temp.name

        with st.spinner("Running vision analysis, preliminary risk scoring, SOP retrieval, and report generation..."):
            state = run_incident_workflow(temp_path, hazard_type, location, user_notes)

        st.session_state["incident_result"] = {
            "state": state,
            "image_bytes": uploaded.getvalue(),
            "filename": uploaded.name,
        }
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink(missing_ok=True)

if "incident_result" in st.session_state:
    result = st.session_state["incident_result"]
    state = result["state"]
    risk = state.get("risk", {})
    st.divider()
    st.header("Incident dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preliminary risk", risk.get("risk_level", "Unknown"))
    c2.metric("Detected objects", len(state.get("detections", [])))
    c3.metric("Hazard type", hazard_type if "hazard_type" in locals() else "See report")
    c4.metric("Knowledge passages", len(state.get("retrieved_context", [])))

    tabs = st.tabs(["Vision & detections", "Risk estimate", "RAG evidence", "Agent report", "Export"])
    with tabs[0]:
        st.subheader("Visual model observations")
        st.write(state.get("visual_description", "No description available."))
        dets = state.get("detections", [])
        if dets:
            st.dataframe(pd.DataFrame(dets), use_container_width=True, hide_index=True)
        else:
            st.info("No YOLO detections returned. This does not mean the image is hazard-free.")
        annotated = state.get("annotated_image_path")
        if annotated and Path(annotated).exists():
            st.image(annotated, caption="YOLO annotated image", use_container_width=True)
    with tabs[1]:
        st.subheader("Preliminary risk estimate")
        st.write("**Risk level:**", risk.get("risk_level", "Unknown"))
        st.write("**Model confidence (not a calibrated probability):**", risk.get("model_confidence", "N/A"))
        st.write(risk.get("rationale", ""))
        st.json(risk.get("features", {}))
        st.info("The baseline classifier uses a tiny synthetic demonstration dataset. Do not use it for operational safety decisions.")
    with tabs[2]:
        context = state.get("retrieved_context", [])
        if context:
            for i, passage in enumerate(context, 1):
                with st.expander(f"Retrieved passage {i}"):
                    st.write(passage)
        else:
            st.warning("No passages retrieved. Check the knowledge_base folder and embedding dependencies.")
    with tabs[3]:
        st.markdown(state.get("final_report", "No report was generated."))
        for error in state.get("errors", []):
            st.warning(error)
    with tabs[4]:
        report = {
            "source_image": result["filename"],
            "risk": risk,
            "detections": state.get("detections", []),
            "visual_description": state.get("visual_description", ""),
            "retrieved_context": state.get("retrieved_context", []),
            "report": state.get("final_report", ""),
            "errors": state.get("errors", []),
        }
        st.download_button(
            "Download incident report (JSON)",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="road_hazard_incident_report.json",
            mime="application/json",
        )
        st.download_button(
            "Download report (Markdown)",
            data=state.get("final_report", ""),
            file_name="road_hazard_incident_report.md",
            mime="text/markdown",
        )
