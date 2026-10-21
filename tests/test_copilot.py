"""
Week 5 Comprehensive Verification & Dirty Dataset Stress-Test Suite.
Tests Layer 1 (Ingestion, SQLAlchemy SQL URI, & Vector Schema Catalog),
Layer 2 (LangGraph State Machine & Query-Aware Target Selection),
Layer 3 (AST Security Guardrails & Subprocess Sandbox),
Layer 4 (Multimodal Synthesis, Scikit-Learn Feature Importance, & Plotly JSON),
and Dirty Dataset Stress Cases (missing headers, mixed currency types, high missingness, wide tables).
"""
import tempfile
import unittest
import numpy as np
import pandas as pd
import sqlite3

from backend.agent import AutonomousDataAnalystAgent
from backend.ingestion import IngestionAndProfilingEngine
from backend.sandbox import SandboxedExecutionRuntime, validate_code_ast


class TestAutonomousDataAnalystCopilot(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = IngestionAndProfilingEngine()
        self.sandbox = SandboxedExecutionRuntime(timeout_seconds=15, max_memory_mb=512)
        self.agent = AutonomousDataAnalystAgent(self.engine, self.sandbox)

    def test_ast_security_blocks_forbidden_imports_and_builtins(self) -> None:
        malicious_snippets = [
            "import os\nos.system('whoami')",
            "import subprocess\nsubprocess.Popen(['ls'])",
            "from socket import socket",
            "eval('2 + 2')",
            "exec('import sys')",
            "open('/etc/passwd', 'r')",
            "().__class__.__bases__[0].__subclasses__()",
            "raise SystemExit(1)",
            "raise SystemExit",
        ]
        for snippet in malicious_snippets:
            report = validate_code_ast(snippet)
            self.assertFalse(report["is_safe"], f"Failed to block: {snippet}")
            self.assertGreater(len(report["violations"]), 0)

    def test_ast_security_allows_analytical_and_ml_code(self) -> None:
        safe_code = """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
summary = df.groupby('segment')['monthly_arr_usd'].mean()
"""
        report = validate_code_ast(safe_code)
        self.assertTrue(report["is_safe"])
        self.assertEqual(len(report["violations"]), 0)

    def test_dirty_dataset_stress_test_and_multimodal_outputs(self) -> None:
        dirty_df = pd.DataFrame({
            "Unnamed: 0": [1, 2, 3, 4, 5, 6, 7, 8],
            "   ": ["North", "South", None, "East", "North", "West", None, "South"],
            "revenue_mixed": ["$1,200.50", "N/A", "$3,400.00", "2100.75", "error", "$4,500.25", "1800.0", "2950.5"],
            "churn_flag": [0, 1, 0, np.nan, 1, 0, 0, 1],
            "high_missing_col": [np.nan, np.nan, np.nan, 42.0, np.nan, np.nan, 19.5, np.nan],
        })
        for extra_idx in range(25):
            dirty_df[f"metric_col_{extra_idx}"] = np.linspace(1.0, 10.0, 8) * (extra_idx + 1)

        profile = self.engine.ingest_dataframe(
            dirty_df,
            table_name="dirty_stress_dataset",
            description="Dirty real-world stress dataset with missing headers and mixed currency types",
        )
        self.assertEqual(profile["shape"]["rows"], 8)
        self.assertEqual(profile["shape"]["columns"], 30)
        self.assertTrue(all(not c["name"].startswith("Unnamed") for c in profile["columns"]))

        res = self.agent.run(
            query="Analyze revenue_mixed across regions and handle dirty missing values",
            table_name="dirty_stress_dataset",
            simulate_initial_fault=False,
        )
        self.assertEqual(res["attempts"], 1)
        self.assertTrue(res["final_output"]["success"])
        self.assertIsNotNone(res["final_output"]["chart_base64"])
        result_data = res["final_output"]["result_data"]
        self.assertIn("feature_importance", result_data)
        self.assertIn("plotly_spec", result_data)
        self.assertIn("Completed schema-grounded analysis", res["executive_summary"])

    def test_self_healing_closed_loop_recovery(self) -> None:
        df = pd.DataFrame({
            "segment": ["Enterprise", "Enterprise", "Mid-Market", "SMB", "SMB", "Mid-Market"],
            "arr": [15000, 18000, 6200, 1400, 1900, 5800],
            "tickets": [1, 2, 4, 9, 11, 5],
        })
        self.engine.ingest_dataframe(df, table_name="saas_metrics")

        res = self.agent.run(
            query="Compute mean arr by segment",
            table_name="saas_metrics",
            simulate_initial_fault=True,
        )
        self.assertEqual(res["attempts"], 2)
        self.assertFalse(res["execution_history"][0]["success"])
        self.assertIn("KeyError", res["execution_history"][0]["error_type"])
        self.assertTrue(res["execution_history"][1]["success"])
        self.assertTrue(any(n["node"] == "ReflectorNode" for n in res["node_transitions"]))

    def test_sqlalchemy_database_ingestion(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
            uri = f"sqlite:///{tmp_db.name}"
            sample_sql_df = pd.DataFrame({
                "deal_id": ["D1", "D2", "D3", "D4"],
                "industry": ["FinTech", "Health", "FinTech", "Retail"],
                "deal_size_usd": [85000.0, 120000.0, 95000.0, 45000.0],
                "cycle_days": [34, 62, 40, 28],
            })
            with sqlite3.connect(tmp_db.name) as conn:
                sample_sql_df.to_sql("pipeline_deals", conn, index=False, if_exists="replace")

            profile = self.engine.ingest_sql_uri(
                connection_uri=uri,
                sql_query="SELECT * FROM pipeline_deals",
                table_name="sql_pipeline_deals",
            )
            self.assertEqual(profile["shape"]["rows"], 4)
            self.assertIn("sql_pipeline_deals", self.engine.datasets)


if __name__ == "__main__":
    unittest.main()
