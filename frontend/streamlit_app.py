"""
Week 5 Deliverable: Streamlit Web Interface for Autonomous Data Analyst Copilot.
Includes dataset uploaders, interactive schema profiler, AST security verifier,
LangGraph execution trace, and multimodal chart/narrative output.
"""
import base64
import pandas as pd
import streamlit as st
from backend.agent import AutonomousDataAnalystAgent
from backend.ingestion import IngestionAndProfilingEngine
from backend.main import seed_default_datasets, engine, agent
from backend.sandbox import validate_code_ast

st.set_page_config(
    page_title="Autonomous Data Analyst Copilot",
    page_icon="📊",
    layout="wide",
)

st.title("Autonomous Data Analyst Copilot")
st.caption("Multimodal Schema RAG • LangGraph State Machine • AST Security Sandbox • Self-Healing Runtime")

with st.sidebar:
    st.header("1. Data Ingestion")
    uploaded = st.file_uploader("Upload CSV or Excel (.xlsx)", type=["csv", "xlsx"])
    if uploaded is not None:
        profile = engine.ingest_file_bytes(uploaded.getvalue(), uploaded.name)
        st.success(f"Ingested `{profile['table_name']}` ({profile['shape']['rows']} rows)")

    available_tables = list(engine.datasets.keys())
    selected_table = st.selectbox("Active Dataset", available_tables, index=0)
    simulate_fault = st.checkbox("Simulate Dirty Data / Self-Healing Cycle", value=False)

    st.header("2. AST Security Guardrail")
    sample_ast_code = st.text_area(
        "Test Code Snippet against AST Policy",
        value="import pandas as pd\nresult = df.describe()",
        height=110,
    )
    if st.button("Validate AST"):
        rep = validate_code_ast(sample_ast_code)
        if rep["is_safe"]:
            st.success(f"SAFE ({rep['ast_node_count']} AST nodes verified)")
        else:
            st.error("BLOCKED: " + "; ".join(rep["violations"]))

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Schema Profile: `{selected_table}`")
    schema = engine.catalog.schemas[selected_table]
    st.json({
        "shape": schema["shape"],
        "memory_kb": schema["memory_kb"],
        "columns": [
            {
                "name": c["name"],
                "dtype": c["dtype"],
                "missing_ratio": c["missing_ratio"],
                "cardinality": c["cardinality"],
            }
            for c in schema["columns"]
        ],
    })
    st.dataframe(engine.datasets[selected_table].head(10), use_container_width=True)

with col2:
    st.subheader("Agentic Query & Multimodal Synthesis")
    user_query = st.text_input(
        "Ask a business question:",
        value="What factors contribute to customer churn across segments?",
    )
    if st.button("Run Autonomous Analysis", type="primary"):
        res = agent.run(
            query=user_query,
            table_name=selected_table,
            simulate_initial_fault=simulate_fault,
        )
        st.markdown("### Executive Narrative Summary")
        st.info(res["executive_summary"])

        st.markdown("### LangGraph State Transitions")
        st.table(pd.DataFrame(res["node_transitions"]))

        if res["final_output"] and res["final_output"].get("chart_base64"):
            st.markdown("### Generated Visualization Artifact")
            img_bytes = base64.b64decode(res["final_output"]["chart_base64"])
            st.image(img_bytes, use_container_width=True)

        with st.expander("View Executed Sandboxed Code & Output"):
            st.code(res["final_code"], language="python")
            st.json(res["final_output"]["result_data"])
