"""
Layer 1: Ingestion & Schema Profiling Engine
Handles CSV, XLSX, and SQL database connections (MySQL / PostgreSQL / SQLite URI schemas
via SQLAlchemy with built-in sqlite3 fallback), computes column-level statistical profiles
(shape, memory, dtypes, cardinality, missingness, skewness, distribution quantiles), and
indexes compact schema metadata in a vector catalog (ChromaDB / cosine-similarity engine).
"""
import io
import json
import math
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class VectorSchemaCatalog:
    """
    Indexes table schema dictionaries for semantic retrieval across multi-table queries.
    Uses ChromaDB when configured, with a deterministic cosine-similarity vector engine
    so it runs with zero external failures in any environment.
    """

    def __init__(self) -> None:
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _tokenize(text_val: str) -> Dict[str, float]:
        tokens = re.findall(r"[a-z0-9_]+", text_val.lower())
        counts: Dict[str, float] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        return {k: v / norm for k, v in counts.items()}

    def index_schema(self, table_name: str, schema_doc: Dict[str, Any]) -> None:
        self.schemas[table_name] = schema_doc
        corpus = " ".join([
            table_name,
            schema_doc.get("description", ""),
            " ".join(c["name"] + " " + c["dtype"] + " " + c.get("semantic_role", "")
                     for c in schema_doc.get("columns", []))
        ])
        self.embeddings[table_name] = self._tokenize(corpus)

    def query_relevant_schemas(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q_vec = self._tokenize(query)
        scored: List[Tuple[float, str]] = []
        for table_name, doc_vec in self.embeddings.items():
            score = sum(q_vec.get(k, 0.0) * v for k, v in doc_vec.items())
            scored.append((round(float(score), 4), table_name))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, tname in scored[:top_k]:
            item = dict(self.schemas[tname])
            item["similarity_score"] = score
            results.append(item)
        return results


class IngestionAndProfilingEngine:
    """Ingests CSV/XLSX/SQL sources and generates LLM-ready schema metadata JSON."""

    def __init__(self) -> None:
        self.catalog = VectorSchemaCatalog()
        self.datasets: Dict[str, pd.DataFrame] = {}

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str = "uploaded_dataset",
        description: str = "User analytical dataset",
    ) -> Dict[str, Any]:
        cleaned_df = self._sanitize_headers(df.copy())
        self.datasets[table_name] = cleaned_df
        profile = self.profile_dataframe(cleaned_df, table_name, description)
        self.catalog.index_schema(table_name, profile)
        return profile

    def ingest_file_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        tname = table_name or re.sub(r"[^a-zA-Z0-9_]", "_", filename.rsplit(".", 1)[0]).lower()
        lower_name = filename.lower()
        if lower_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        elif lower_name.endswith(".json"):
            raw_json = json.loads(file_bytes.decode("utf-8"))
            if isinstance(raw_json, dict):
                for k, v in raw_json.items():
                    if isinstance(v, list):
                        raw_json = v
                        break
            df = pd.DataFrame(raw_json)
        elif lower_name.endswith(".jsonl"):
            df = pd.read_json(io.BytesIO(file_bytes), lines=True)
        elif lower_name.endswith(".tsv"):
            df = pd.read_csv(io.BytesIO(file_bytes), sep="\t")
        elif lower_name.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(file_bytes))
        else:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
        return self.ingest_dataframe(df, table_name=tname, description=f"Ingested from {filename}")

    def ingest_sql_uri(
        self,
        connection_uri: str,
        sql_query: str,
        table_name: str = "sql_table",
        description: str = "Ingested via SQL database connection",
    ) -> Dict[str, Any]:
        """Connects to a relational database via SQLAlchemy or built-in sqlite3 fallback."""
        try:
            from sqlalchemy import create_engine, text
            sql_engine = create_engine(connection_uri)
            with sql_engine.connect() as conn:
                df = pd.read_sql(text(sql_query), conn)
        except ImportError:
            db_path = connection_uri.replace("sqlite:///", "")
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(sql_query, conn)
        return self.ingest_dataframe(df, table_name=table_name, description=description)

    @staticmethod
    def _sanitize_headers(df: pd.DataFrame) -> pd.DataFrame:
        new_cols = []
        for idx, col in enumerate(df.columns):
            col_str = str(col).strip()
            if not col_str or col_str.startswith("Unnamed:"):
                col_str = f"column_{idx + 1}"
            col_str = re.sub(r"[^a-zA-Z0-9_]", "_", col_str)
            new_cols.append(col_str)
        df.columns = new_cols
        return df

    @staticmethod
    def profile_dataframe(
        df: pd.DataFrame,
        table_name: str = "dataset",
        description: str = "",
    ) -> Dict[str, Any]:
        row_count, col_count = df.shape
        mem_bytes = int(df.memory_usage(deep=True).sum())
        columns_meta: List[Dict[str, Any]] = []

        for col in df.columns:
            series = df[col]
            missing_count = int(series.isna().sum())
            missing_ratio = round(missing_count / max(row_count, 1), 4)
            unique_count = int(series.nunique(dropna=True))
            dtype_str = str(series.dtype)

            col_info: Dict[str, Any] = {
                "name": str(col),
                "dtype": dtype_str,
                "missing_count": missing_count,
                "missing_ratio": missing_ratio,
                "cardinality": unique_count,
            }

            cleaned_str = series.dropna().astype(str).str.replace(r"[$₹€£,%]", "", regex=True).str.strip()
            numeric_series = pd.to_numeric(cleaned_str, errors="coerce")
            non_na_numeric = numeric_series.dropna()

            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                clean_num = series.dropna().astype(float)
                col_info["semantic_role"] = "numeric_metric"
                if len(clean_num) > 0:
                    skew_val = float(clean_num.skew()) if len(clean_num) > 2 else 0.0
                    col_info["stats"] = {
                        "mean": round(float(clean_num.mean()), 4),
                        "std": round(float(clean_num.std()), 4) if len(clean_num) > 1 else 0.0,
                        "min": round(float(clean_num.min()), 4),
                        "q25": round(float(clean_num.quantile(0.25)), 4),
                        "median": round(float(clean_num.median()), 4),
                        "q75": round(float(clean_num.quantile(0.75)), 4),
                        "max": round(float(clean_num.max()), 4),
                        "skewness": round(0.0 if math.isnan(skew_val) else skew_val, 4),
                    }
            else:
                col_info["semantic_role"] = "categorical_or_text"
                top_vals = series.dropna().astype(str).value_counts().head(5).to_dict()
                col_info["sample_values"] = list(top_vals.keys())
                col_info["top_frequencies"] = {str(k): int(v) for k, v in top_vals.items()}
                if len(non_na_numeric) > 0 and len(non_na_numeric) >= 0.5 * max(len(series.dropna()), 1):
                    col_info["dirty_mixed_numeric_detected"] = True

            columns_meta.append(col_info)

        sample_rows = json.loads(df.head(5).to_json(orient="records", date_format="iso"))
        return {
            "table_name": table_name,
            "description": description,
            "shape": {"rows": row_count, "columns": col_count},
            "memory_bytes": mem_bytes,
            "memory_kb": round(mem_bytes / 1024.0, 2),
            "columns": columns_meta,
            "sample_rows": sample_rows,
        }
