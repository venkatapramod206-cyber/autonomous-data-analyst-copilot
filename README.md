---
title: Autonomous Data Analyst Copilot
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
---

# Autonomous Data Analyst Copilot (Multimodal RAG & Dynamic Code Sandbox)

An enterprise-grade agentic data analytics system bridging unstructured business questions and complex tabular data processing across four isolated architectural layers:

1. **Layer 1 — Ingestion & Schema Profiling Engine (`backend/ingestion.py`)**:
   - Ingests `.csv`, `.xlsx`, and SQL tables.
   - Computes automated EDA profiles (shape, memory footprint, dtypes, cardinality, missingness ratios, skewness, and quantiles).
   - Indexes schema metadata into a vector catalog (`VectorSchemaCatalog`) for semantic multi-table retrieval.
2. **Layer 2 — Query Parsing & Agentic Reasoning Engine (`backend/agent.py`)**:
   - Deterministic cyclic state machine (`PlannerNode` -> `CodeGeneratorNode` -> `ExecutionNode` -> `ReflectorNode` -> `InsightSynthesizerNode`).
3. **Layer 3 — Secure Code Generation & Sandboxed Execution Runtime (`backend/sandbox.py`)**:
   - Pre-execution Python `ast.NodeVisitor` guardrail blocking `os`, `sys`, `subprocess`, `socket`, `eval()`, `exec()`, `open()`, and dunder escapes.
   - Isolated subprocess runtime with 10-second CPU timeout and self-healing traceback loop (up to 3 repair attempts).
4. **Layer 4 — Multimodal Synthesis & Visualization Output (`backend/main.py` & `frontend/streamlit_app.py`)**:
   - Captures stdout, structured statistical metrics, and Base64 Seaborn/Matplotlib figures alongside executive narrative summaries.

## Quickstart

```bash
# Run verification suite
pytest tests/test_copilot.py -v

# Start FastAPI server
uvicorn backend.main:app --reload --port 8000

# Start Streamlit UI
streamlit run frontend/streamlit_app.py --server.port 7860
```
