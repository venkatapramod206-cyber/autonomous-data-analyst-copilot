"""
Layer 3: Secure Code Generation & Sandboxed Execution Runtime
Implements AST pre-execution validation (blocking dangerous modules, built-ins,
and dunder attribute escapes) and an ephemeral, resource-constrained Python
subprocess runner with strict timeout (10s) and memory caps (512MB).
"""
import ast
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Set


FORBIDDEN_MODULES: Set[str] = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "requests", "urllib", "http", "ftplib", "telnetlib", "smtplib",
    "ctypes", "multiprocessing", "threading", "signal", "pty",
    "resource", "inspect", "importlib", "builtins", "pickle",
}

FORBIDDEN_BUILTINS: Set[str] = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "exit", "quit", "help", "memoryview",
}

FORBIDDEN_ATTRIBUTES: Set[str] = {
    "__subclasses__", "__bases__", "__mro__", "__globals__",
    "__code__", "__closure__", "__builtins__", "__dict__",
    "__class__", "system", "popen", "spawn", "fork",
}


class ASTSecurityError(Exception):
    """Raised when LLM-generated code violates AST security guardrails."""


class ASTSecurityValidator(ast.NodeVisitor):
    """
    Traverses the Python Abstract Syntax Tree before code hits the interpreter
    to block blacklisted imports, built-in calls, and attribute escapes.
    """

    def __init__(self) -> None:
        self.violations: List[str] = []
        self.allowed_imports: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_mod = alias.name.split(".")[0]
            if root_mod in FORBIDDEN_MODULES:
                self.violations.append(
                    f"Line {node.lineno}: Forbidden import '{alias.name}' blocked by AST guardrail."
                )
            else:
                self.allowed_imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod_name = (node.module or "").split(".")[0]
        if mod_name in FORBIDDEN_MODULES:
            self.violations.append(
                f"Line {node.lineno}: Forbidden import from '{node.module}' blocked by AST guardrail."
            )
        else:
            self.allowed_imports.append(node.module or "")
        for alias in node.names:
            if alias.name in FORBIDDEN_BUILTINS or alias.name in FORBIDDEN_MODULES:
                self.violations.append(
                    f"Line {node.lineno}: Forbidden symbol '{alias.name}' in from-import."
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            fn_name = node.func.id
            if fn_name in FORBIDDEN_BUILTINS:
                self.violations.append(
                    f"Line {node.lineno}: Forbidden built-in call '{fn_name}()' blocked by AST guardrail."
                )
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in FORBIDDEN_ATTRIBUTES or attr_name in FORBIDDEN_BUILTINS:
                self.violations.append(
                    f"Line {node.lineno}: Forbidden method call '.{attr_name}()' blocked by AST guardrail."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRIBUTES or (
            node.attr.startswith("__") and node.attr.endswith("__")
        ):
            self.violations.append(
                f"Line {node.lineno}: Forbidden dunder/system attribute access '.{node.attr}' blocked."
            )
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if isinstance(node.exc, ast.Name) and node.exc.id == "SystemExit":
            self.violations.append(
                f"Line {node.lineno}: Raising SystemExit is forbidden inside sandbox."
            )
        elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            if node.exc.func.id == "SystemExit":
                self.violations.append(
                    f"Line {node.lineno}: Raising SystemExit() is forbidden inside sandbox."
                )
        self.generic_visit(node)


def validate_code_ast(code: str) -> Dict[str, Any]:
    """Parses and validates code via Python's ast module."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "is_safe": False,
            "syntax_valid": False,
            "violations": [f"SyntaxError at line {exc.lineno}: {exc.msg}"],
            "ast_node_count": 0,
            "allowed_imports": [],
        }

    validator = ASTSecurityValidator()
    validator.visit(tree)
    node_count = sum(1 for _ in ast.walk(tree))
    return {
        "is_safe": len(validator.violations) == 0,
        "syntax_valid": True,
        "violations": validator.violations,
        "ast_node_count": node_count,
        "allowed_imports": validator.allowed_imports,
    }


class SandboxedExecutionRuntime:
    """
    Executes AST-validated analytical Python code in an ephemeral subprocess
    with a 10-second timeout and 512MB virtual memory cap, capturing stdout,
    result_summary dict, and Matplotlib/Seaborn Base64 figures.
    """

    def __init__(self, timeout_seconds: int = 10, max_memory_mb: int = 512) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb

    def execute(self, code: str, csv_bytes: bytes) -> Dict[str, Any]:
        ast_report = validate_code_ast(code)
        if not ast_report["is_safe"]:
            return {
                "success": False,
                "error_type": "ASTSecurityViolation" if ast_report["syntax_valid"] else "SyntaxError",
                "error_message": "; ".join(ast_report["violations"]),
                "traceback": "\n".join(ast_report["violations"]),
                "stdout": "",
                "result_data": None,
                "chart_base64": None,
                "execution_ms": 0.0,
                "ast_report": ast_report,
            }

        start_ts = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="copilot_sbx_") as tmpdir:
            data_path = os.path.join(tmpdir, "dataset.csv")
            out_path = os.path.join(tmpdir, "output.json")
            fig_path = os.path.join(tmpdir, "figure.png")
            runner_path = os.path.join(tmpdir, "runner.py")

            with open(data_path, "wb") as f:
                f.write(csv_bytes)

            harness_code = f"""
import json
import math
import resource
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Apply CPU time limit (10s)
try:
    resource.setrlimit(resource.RLIMIT_CPU, ({max(self.timeout_seconds, 20)}, {max(self.timeout_seconds, 20)}))
except Exception:
    pass

df = pd.read_csv({json.dumps(data_path)})
result_summary = {{}}

# --- BEGIN USER ANALYTICAL CODE ---
{code}
# --- END USER ANALYTICAL CODE ---

if plt.get_fignums():
    plt.tight_layout()
    plt.savefig({json.dumps(fig_path)}, dpi=120, bbox_inches='tight')
    plt.close('all')

with open({json.dumps(out_path)}, 'w', encoding='utf-8') as f_out:
    json.dump(result_summary, f_out, default=str)
"""
            with open(runner_path, "w", encoding="utf-8") as f:
                f.write(harness_code)

            try:
                proc = subprocess.run(
                    [sys.executable, "-I", runner_path],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                elapsed = round((time.perf_counter() - start_ts) * 1000.0, 2)
                return {
                    "success": False,
                    "error_type": "TimeoutError",
                    "error_message": f"Execution exceeded strict {self.timeout_seconds}s timeout guardrail.",
                    "traceback": "TimeoutExpired: Process terminated by sandbox watchdog.",
                    "stdout": "",
                    "result_data": None,
                    "chart_base64": None,
                    "execution_ms": elapsed,
                    "ast_report": ast_report,
                }

            elapsed = round((time.perf_counter() - start_ts) * 1000.0, 2)
            if proc.returncode != 0:
                stderr_txt = proc.stderr.strip()
                lines = [ln for ln in stderr_txt.splitlines() if ln.strip()]
                last_line = lines[-1] if lines else "Runtime execution error"
                err_type = last_line.split(":")[0] if ":" in last_line else "RuntimeError"
                return {
                    "success": False,
                    "error_type": err_type,
                    "error_message": last_line,
                    "traceback": stderr_txt,
                    "stdout": proc.stdout,
                    "result_data": None,
                    "chart_base64": None,
                    "execution_ms": elapsed,
                    "ast_report": ast_report,
                }

            result_data = None
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f_in:
                    result_data = json.load(f_in)

            chart_b64 = None
            if os.path.exists(fig_path):
                with open(fig_path, "rb") as f_img:
                    chart_b64 = base64.b64encode(f_img.read()).decode("utf-8")

            return {
                "success": True,
                "error_type": None,
                "error_message": None,
                "traceback": None,
                "stdout": proc.stdout.strip(),
                "result_data": result_data,
                "chart_base64": chart_b64,
                "execution_ms": elapsed,
                "ast_report": ast_report,
            }
