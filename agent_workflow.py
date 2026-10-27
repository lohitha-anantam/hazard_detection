"""LangGraph orchestration: vision observations -> risk estimate -> RAG -> recommendations."""
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

class IncidentState(TypedDict, total=False):
    image_path: str
    hazard_type: str
    location: str
    user_notes: str
    detections: list[dict]
    annotated_image_path: str
    visual_description: str
    risk: dict
    retrieved_context: list[str]
    final_report: str
    errors: list[str]


def build_workflow():
    def vision_node(state: IncidentState):
        from vision_engine import detect_objects, interpret_image_with_vision_model
        errors = list(state.get("errors", []))
        try:
            detections, annotated = detect_objects(state["image_path"])
        except Exception as exc:
            detections, annotated = [], ""
            errors.append(f"YOLO detection: {exc}")
        try:
            description = interpret_image_with_vision_model(state["image_path"])
        except Exception as exc:
            description = "Vision model analysis failed. Please review the image manually."
            errors.append(f"Vision model: {exc}")
        return {
            "detections": detections,
            "annotated_image_path": annotated,
            "visual_description": description,
            "errors": errors,
        }

    def risk_node(state: IncidentState):
        from risk_engine import score_incident
        text = state.get("visual_description", "") + " " + state.get("user_notes", "")
        risk = score_incident(state.get("detections", []), text, state.get("hazard_type", "Unknown"))
        return {"risk": risk}

    def retrieval_node(state: IncidentState):
        from rag_engine import retrieve_policy_context
        query = (
            f"{state.get('hazard_type', '')} {state.get('visual_description', '')} "
            f"{state.get('risk', {}).get('risk_level', '')} road response"
        )
        try:
            context = retrieve_policy_context(query)
            return {"retrieved_context": context}
        except Exception as exc:
            return {
                "retrieved_context": [],
                "errors": state.get("errors", []) + [f"RAG retrieval: {exc}"],
            }

    def report_node(state: IncidentState):
        from report_engine import generate_report
        try:
            report = generate_report(state)
        except Exception as exc:
            report = (
                "LLM report generation failed. Human review is required.\n\n"
                f"Visual observations: {state.get('visual_description', 'Unavailable')}\n"
                f"Risk estimate: {state.get('risk', {})}\n"
                f"Retrieved context: {' '.join(state.get('retrieved_context', []))}"
            )
            return {
                "final_report": report,
                "errors": state.get("errors", []) + [f"Report generation: {exc}"],
            }
        return {"final_report": report}

    graph = StateGraph(IncidentState)
    graph.add_node("vision_analysis", vision_node)
    graph.add_node("risk_estimation", risk_node)
    graph.add_node("policy_retrieval", retrieval_node)
    graph.add_node("report_generation", report_node)
    graph.set_entry_point("vision_analysis")
    graph.add_edge("vision_analysis", "risk_estimation")
    graph.add_edge("risk_estimation", "policy_retrieval")
    graph.add_edge("policy_retrieval", "report_generation")
    graph.add_edge("report_generation", END)
    return graph.compile()


def run_incident_workflow(image_path: str, hazard_type: str, location: str, user_notes: str):
    return build_workflow().invoke({
        "image_path": image_path,
        "hazard_type": hazard_type,
        "location": location,
        "user_notes": user_notes,
        "errors": [],
    })
