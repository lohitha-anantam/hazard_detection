"""Computer-vision utilities: YOLO detection and optional OpenRouter image interpretation."""
from pathlib import Path
import base64
import mimetypes

import cv2
from PIL import Image
from ultralytics import YOLO

from config import YOLO_MODEL, YOLO_CONFIDENCE, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_VISION_MODEL


def detect_objects(image_path: str, confidence: float = YOLO_CONFIDENCE):
    """Detect common COCO objects in one image. This is not a trained pothole detector."""
    model = YOLO(YOLO_MODEL)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("OpenCV could not read the uploaded image.")
    result = model.predict(image, conf=confidence, verbose=False)[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            detections.append({
                "label": result.names.get(cls_id, str(cls_id)),
                "confidence": round(float(box.conf[0].item()), 3),
                "box_xyxy": [int(v) for v in box.xyxy[0].tolist()],
            })
    annotated = result.plot()
    annotated_path = Path(image_path).with_name(Path(image_path).stem + "_detections.jpg")
    cv2.imwrite(str(annotated_path), annotated)
    return detections, str(annotated_path)


def interpret_image_with_vision_model(image_path: str) -> str:
    """Ask an OpenRouter vision-capable model for cautious visual observations."""
    if not OPENROUTER_API_KEY:
        return (
            "Vision LLM analysis was skipped because OPENROUTER_API_KEY is not configured. "
            "Use the YOLO detections as limited object observations and describe the hazard manually."
        )

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    image_bytes = Path(image_path).read_bytes()
    data_url = f"data:{mime};base64," + base64.b64encode(image_bytes).decode("utf-8")
    llm = ChatOpenAI(
        model=OPENROUTER_VISION_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        temperature=0,
    )
    message = HumanMessage(content=[
        {"type": "text", "text": (
            "You are assisting a human road-maintenance operator. Describe only visible evidence "
            "relevant to road hazards: apparent hazard type, visible location in the image, possible "
            "impact on road users, and uncertainty. Do not infer exact dimensions, injury, cause, "
            "legal responsibility, or facts outside the image. This is preliminary triage, not an "
            "official determination. If no road hazard is visible, say so."
        )},
        {"type": "image_url", "image_url": {"url": data_url}},
    ])
    response = llm.invoke([message])
    return str(response.content)
