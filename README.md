# SIH26003: AI-Based Cognitive Gaming and Memory Assistance Platform

## 1. Project Overview
This repository contains the integrated prototype for the SIH26003 problem statement: AI-Based Cognitive Gaming and Memory Assistance Platform for Elderly Dementia Patients in the North Eastern Region (NER).

The repository has been structured cleanly into modular domains:
- **Backend:** Central FastAPI server uniting all analytics and AI capabilities.
- **ML Analytics:** Processes telemetry to provide baseline comparisons and behavioral scoring.
- **Cultural AI:** Uses an LLM to generate culturally grounded reminiscence scenarios (specific to the NER).
- **Dashboard:** A Streamlit UI providing a unified view for evaluators.

## 2. Backend
The `backend/` module hosts a FastAPI application (`backend/main.py`). It exposes API endpoints that:
- Handle interaction telemetry and logging into MySQL.
- Route ML evaluation data to the ML pipeline.
- Route patient profile requests to the Cultural AI scenario generator.

## 3. ML Analytics
The `ml-analytics/` module is a standalone pandas-based data pipeline. It normalizes input telemetry (such as touch drift, decision speed, response latency), compares against a user's historical baseline, and generates behavioral indicator trends. It never diagnoses dementia, rather outputting a neutral observation score.

## 4. Cultural AI
The `ai-cultural-narrative/` module provides a respectful reminiscence scenario generation module. Grounded on an NER cultural dataset (`ai-cultural-narrative/data/ner_culture.json`), it formulates custom follow-up questions securely avoiding unsupported regions or festivals.

## 5. Dashboard
The `dashboard/` module holds a Streamlit-based interface used to visualize the processed ML metrics and cultural narratives via radar charts and engagement progression tracks. The dashboard relies entirely on the Backend APIs (never running raw analytic models itself).

## 6. How the Modules Connect
1. The **Dashboard** retrieves telemetry and cultural narratives via HTTP calls (via `dashboard/api_client.py`).
2. The **Backend** receives the HTTP requests from the Dashboard.
3. The **Backend** directly imports and runs `ml-analytics` code (via `sys.path`) to evaluate behavioral telemetry and passes back JSON results.
4. The **Backend** imports and evaluates `ai-cultural-narrative` logic to produce culturally-relevant dialogue schemas.

## 7. Installation
Make sure you have Python 3.9+ installed. From the repository root, install the unified dependencies:
```bash
pip3 install -r requirements.txt
```

Set up your `.env` file (based on `.env.example`):
```bash
cp .env.example .env
```
Ensure you have MySQL running if logging telemetry, and populate `LLM_API_KEY` for real generative text (the system uses an offline demo stub if the key is missing).

## 8. How to run Backend
Start the FastAPI server from the repository root:
```bash
uvicorn backend.main:app --reload
```
It will be available at `http://127.0.0.1:8000`.

## 9. How to run Dashboard
Open a separate terminal, activate the environment, and from the repository root:
```bash
streamlit run dashboard/app.py
```
It will start on port `8501`.

## 10. How to run Tests
**ML Tests:**
```bash
cd ml-analytics
python3 tests/test_pipeline.py
```

**Cultural AI Tests:**
```bash
cd ai-cultural-narrative
PYTHONPATH=$(pwd)/src python3 tests/test_scenarios.py
```
