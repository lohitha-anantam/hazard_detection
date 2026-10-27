"""Generate evidence-grounded incident triage reports."""
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_TEXT_MODEL


def generate_report(state: dict) -> str:
    context = "\n\n".join(state.get("retrieved_context", [])) or "No relevant SOP context was retrieved."
    payload = f"""
Reporter-supplied hazard type: {state.get('hazard_type', 'Unknown')}
Reporter-supplied location: {state.get('location', 'Not supplied')}
Reporter notes: {state.get('user_notes', 'None')}
Visual model observations: {state.get('visual_description', 'Unavailable')}
YOLO detections: {state.get('detections', [])}
Preliminary risk estimate: {state.get('risk', {})}
Retrieved knowledge-base context:
{context}
"""
    if not OPENROUTER_API_KEY:
        return (
            "## Preliminary incident triage (LLM not configured)\n\n"
            + payload
            + "\n\n## Required next steps\n"
            "- A human reviewer must verify the image and risk estimate.\n"
            "- If there is immediate danger to life, injury, fire, or an active collision, contact local emergency services.\n"
            "- This application does not dispatch emergency services or create an official work order."
        )

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=OPENROUTER_TEXT_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        temperature=0.1,
    )
    prompt = f"""
You are an assistant for a human road-maintenance incident reviewer.
Produce a concise structured preliminary triage report with:
1. Observed evidence (separate computer-vision detections from user-provided statements)
2. What is uncertain / cannot be concluded
3. Preliminary severity and why, treating the provided risk estimate as unvalidated
4. Suggested next steps based on the retrieved guide
5. A draft work-order summary with the reported location
6. A human verification checklist

Use only the supplied evidence and context. Do not invent injury, exact dimensions, cause, legal fault, or official policy.
Do not claim to dispatch responders. If immediate danger is reported, advise contacting local emergency services.
Clearly state this is not an official safety decision.

Incident data:
{payload}
"""
    response = llm.invoke(prompt)
    return str(response.content)
