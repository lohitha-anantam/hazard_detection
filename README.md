# Road Hazard Detection & Response Copilot

A practical decision-support prototype that combines computer vision, a baseline ML risk estimator, retrieval-augmented generation (RAG), an LLM, and LangGraph orchestration to help a human reviewer triage road-hazard reports.

## Why this project?

Road damage, debris, damaged manholes, and blocked roads can create safety and maintenance problems. Reports may be inconsistent and maintenance teams need structured evidence and clear handoff notes. This app creates a preliminary report from an uploaded image, reporter-provided details, object detections, and a local response guide.

## Technology stack

- **Computer vision / deep learning:** Ultralytics YOLO detects common objects (people, cars, trucks, buses, bicycles, etc.) and creates an annotated image.
- **Vision LLM / GenAI:** Optional OpenRouter vision model describes visible road-related evidence cautiously.
- **ML:** Scikit-learn RandomForest baseline produces a low/medium/high triage estimate from detections and text keywords.
- **RAG:** Sentence Transformers embeddings and FAISS retrieve passages from local Markdown, text, or PDF guidance in `knowledge_base/`.
- **Agentic orchestration:** LangGraph runs vision analysis, risk estimation, policy retrieval, and report generation as distinct nodes.
- **LLM:** OpenRouter chat model generates a structured preliminary incident report and draft work-order summary.
- **UI:** Streamlit dashboard with evidence, risk features, retrieved passages, and JSON/Markdown export.

## Project structure

```text
road_hazard_response_copilot/
├── app.py
├── agent_workflow.py
├── vision_engine.py
├── risk_engine.py
├── rag_engine.py
├── report_engine.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── knowledge_base/
│   └── road_response_sop.md
├── data/
└── reports/
```

## Run in VS Code (Windows)

### 1. Extract and open the folder

Open `road_hazard_response_copilot` in VS Code.

### 2. Create a virtual environment and install packages

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell activation is blocked, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Configure OpenRouter (optional)

Copy `.env.example` to `.env` and add your own API key. Keep `.env` private and never upload it to GitHub.

- Without an API key, the app still runs YOLO detection and local RAG; vision-LLM interpretation is skipped and a fallback report is provided.
- With an API key, use a vision-capable model available to your account for image interpretation. Set `OPENROUTER_VISION_MODEL` to a supported model ID.
- `OPENROUTER_TEXT_MODEL` must also be a model available to your account. The `openrouter/free` route may not always be available.

### 4. Start the app

```bash
streamlit run app.py
```

Upload a road image, select a reported hazard category, enter the location and notes, and click **Analyze incident**.

The first run may download YOLO weights and the Sentence Transformers embedding model. Internet access is needed for those downloads and for OpenRouter requests.

## RAG knowledge base

Put approved local SOPs or public guidance documents (`.md`, `.txt`, or `.pdf`) into `knowledge_base/`. The app chunks and embeds these files, then retrieves passages relevant to the incident. The included guide is a demo, not an official standard. Replace it with approved local guidance before real-world use.

## Important limitations and responsible use

- Standard YOLO weights are **not a pothole/road-damage detector**. They detect common objects; the optional vision LLM can describe images but may hallucinate. For reliable pothole detection, train/evaluate a dedicated model on a labelled road-damage dataset.
- The ML classifier is trained on a tiny synthetic example dataset solely to demonstrate the pipeline. It is not validated and must not determine real-world response priority.
- No emergency services are contacted, no crew is dispatched, and no official work order is created.
- A human must verify image interpretation, severity, location, and all generated text.
- Do not approach traffic to take pictures. Avoid uploading identifiable people or license plates without appropriate permission.
- This app is a prototype, not a certified emergency or public-safety system.

## GitHub safety

Do not commit `.env`, API keys, private incident images, model weights, or reports containing personal information. `.gitignore` excludes local environment files and common generated assets.
