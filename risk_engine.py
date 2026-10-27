"""Transparent baseline ML-style risk scoring using interpretable features."""
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Synthetic demonstration data only; not a validated safety model.
# Features: people detected, vehicles detected, visible obstruction keyword, road damage keyword.
_TRAIN_X = np.array([
    [0, 0, 0, 0], [0, 2, 0, 0], [1, 1, 0, 0], [2, 5, 1, 0],
    [0, 4, 1, 1], [3, 7, 1, 1], [0, 1, 0, 1], [4, 8, 1, 1],
    [1, 0, 1, 0], [0, 0, 0, 1], [2, 3, 0, 1], [5, 9, 1, 0],
    [0, 6, 0, 0], [1, 6, 1, 1], [0, 0, 1, 1], [3, 2, 1, 1],
], dtype=float)
_TRAIN_Y = np.array([0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2])


def score_incident(detections, visual_description: str, user_hazard_type: str):
    """Return a cautious demo triage score. Human review is mandatory."""
    labels = [d["label"].lower() for d in detections]
    people = sum(label == "person" for label in labels)
    vehicles = sum(label in {"car", "truck", "bus", "motorcycle", "bicycle"} for label in labels)
    text = f"{visual_description} {user_hazard_type}".lower()
    obstruction = int(any(w in text for w in ["blocked", "obstruction", "debris", "fallen", "blocked lane"]))
    road_damage = int(any(w in text for w in ["pothole", "crack", "damaged road", "sinkhole", "manhole"]))
    features = np.array([[min(people, 10), min(vehicles, 15), obstruction, road_damage]], dtype=float)

    model = RandomForestClassifier(n_estimators=80, random_state=42, max_depth=4)
    model.fit(_TRAIN_X, _TRAIN_Y)
    predicted_class = int(model.predict(features)[0])
    probs = model.predict_proba(features)[0]
    class_to_name = {0: "Low", 1: "Medium", 2: "High"}
    probability = float(max(probs))

    # Conservative override: model never decides emergency response.
    urgent_words = ["collision in progress", "injury", "fire", "person trapped", "life-threatening"]
    if any(w in text for w in urgent_words):
        predicted_class = 2
        rationale = "Urgent wording was supplied; immediate human/emergency review is required."
    else:
        rationale = (
            "Baseline score uses a small synthetic demonstration dataset and detected-object/text features. "
            "It is not validated for operational safety decisions."
        )

    return {
        "risk_level": class_to_name[predicted_class],
        "model_confidence": round(probability, 3),
        "features": {
            "people_detected": people,
            "vehicles_detected": vehicles,
            "obstruction_keyword_detected": obstruction,
            "road_damage_keyword_detected": road_damage,
        },
        "rationale": rationale,
    }
