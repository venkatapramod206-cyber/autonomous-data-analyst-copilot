"""
FastAPI REST & WebSocket Backend for Autonomous Data Analyst Copilot.
Exposes endpoints for dataset upload, schema profiling, AST security verification,
synchronous/streaming agent execution, and health monitoring.
"""
from typing import Any, Dict, Optional
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent import AutonomousDataAnalystAgent
from .ingestion import IngestionAndProfilingEngine
from .sandbox import validate_code_ast

app = FastAPI(
    title="Autonomous Data Analyst Copilot API",
    description="Multimodal RAG & Dynamic AST-Sandboxed Code Execution Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = IngestionAndProfilingEngine()
agent = AutonomousDataAnalystAgent(ingestion_engine=engine)


def seed_default_datasets() -> None:
    """Seeds enterprise SaaS churn and dirty stress-test datasets on startup."""
    saas_df = pd.DataFrame({
        "customer_id": [f"CUST-{1000 + i}" for i in range(40)],
        "segment": (["Enterprise", "Mid-Market", "SMB", "Enterprise", "SMB"] * 8),
        "monthly_arr_usd": [
            14200, 4800, 1150, 18900, 920, 15400, 5100, 1340, 22400, 890,
            13100, 4600, 1280, 19800, 1050, 16200, 5300, 1410, 21000, 970,
            14800, 4950, 1190, 17600, 940, 15900, 5250, 1380, 23100, 910,
            13900, 4700, 1220, 20100, 1010, 16800, 5400, 1450, 21800, 960,
        ],
        "support_tickets_90d": [1, 4, 9, 2, 11, 1, 3, 8, 2, 12, 2, 5, 7, 1, 10, 2, 4, 9, 1, 11] * 2,
        "usage_score": [88.5, 71.2, 42.0, 92.1, 38.4, 85.0, 74.5, 49.2, 94.0, 35.1] * 4,
        "churn_risk_score": [0.11, 0.34, 0.78, 0.08, 0.84, 0.14, 0.29, 0.72, 0.06, 0.88] * 4,
    })
    engine.ingest_dataframe(
        saas_df,
        table_name="enterprise_saas_churn",
        description="Enterprise SaaS customer retention, ARR, support ticket volume, and churn risk metrics.",
    )


seed_default_datasets()


class QueryRequest(BaseModel):
    query: str
    table_name: str = "enterprise_saas_churn"
    simulate_initial_fault: bool = False


class ASTCheckRequest(BaseModel):
    code: str


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "autonomous-data-analyst-copilot",
        "indexed_tables": list(engine.datasets.keys()),
    }


@app.get("/schemas")
def list_schemas(query: Optional[str] = None) -> Dict[str, Any]:
    if query:
        return {"schemas": engine.catalog.query_relevant_schemas(query, top_k=5)}
    return {"schemas": list(engine.catalog.schemas.values())}


@app.post("/ingest")
async def ingest_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    raw_bytes = await file.read()
    try:
        profile = engine.ingest_file_bytes(raw_bytes, file.filename or "dataset.csv")
        return {"status": "ingested", "profile": profile}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}") from exc


@app.post("/ast/validate")
def validate_ast_endpoint(req: ASTCheckRequest) -> Dict[str, Any]:
    return validate_code_ast(req.code)


@app.post("/analyze")
def analyze_query(req: QueryRequest) -> Dict[str, Any]:
    if req.table_name not in engine.datasets:
        raise HTTPException(status_code=404, detail=f"Table '{req.table_name}' not found.")
    return agent.run(
        query=req.query,
        table_name=req.table_name,
        simulate_initial_fault=req.simulate_initial_fault,
    )


@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        query = payload.get("query", "Analyze key drivers and segment breakdown")
        table_name = payload.get("table_name", "enterprise_saas_churn")
        simulate_fault = bool(payload.get("simulate_initial_fault", False))
        result = agent.run(query=query, table_name=table_name, simulate_initial_fault=simulate_fault)
        for step in result["node_transitions"]:
            await websocket.send_json({"event": "node_transition", "data": step})
        await websocket.send_json({"event": "complete", "data": result})
    except WebSocketDisconnect:
        pass
