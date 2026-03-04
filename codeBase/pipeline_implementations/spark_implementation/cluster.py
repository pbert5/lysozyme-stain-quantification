from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping, Optional


def _resolve_master(*, master: Optional[str], n_workers: Optional[int]) -> str:
    if master and str(master).strip():
        return str(master).strip()
    if n_workers is not None and int(n_workers) > 0:
        return f"local[{int(n_workers)}]"
    return "local[*]"


def _set_default_env(var: str, value: str) -> None:
    if not os.environ.get(var):
        os.environ[var] = value


def create_local_spark_session(
    *,
    app_name: str = "lysozyme-spark-pipeline",
    master: Optional[str] = None,
    n_workers: Optional[int] = None,
    spark_config: Optional[Mapping[str, str]] = None,
    log_level: str = "WARN",
):
    """
    Create and configure a local SparkSession.

    Raises RuntimeError with a clear dependency hint when pyspark is unavailable.
    """
    try:
        from pyspark.sql import SparkSession  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-driven
        raise RuntimeError(
            "pyspark is required for the spark backend. Install it with `pip install pyspark`."
        ) from exc

    effective_master = _resolve_master(master=master, n_workers=n_workers)
    venv_python = Path(sys.prefix) / "bin" / "python"
    python_exec = str(venv_python if venv_python.exists() else Path(sys.executable).resolve())
    os.environ["PYSPARK_PYTHON"] = python_exec
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec
    _set_default_env("OMP_NUM_THREADS", "1")
    _set_default_env("OPENBLAS_NUM_THREADS", "1")
    _set_default_env("MKL_NUM_THREADS", "1")
    _set_default_env("NUMEXPR_NUM_THREADS", "1")

    builder = SparkSession.builder.appName(app_name).master(effective_master)
    builder = builder.config("spark.pyspark.driver.python", python_exec)
    builder = builder.config("spark.pyspark.python", python_exec)
    builder = builder.config("spark.executorEnv.PYSPARK_PYTHON", python_exec)
    builder = builder.config("spark.executorEnv.PYSPARK_DRIVER_PYTHON", python_exec)
    builder = builder.config("spark.executorEnv.OMP_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
    builder = builder.config("spark.executorEnv.OPENBLAS_NUM_THREADS", os.environ["OPENBLAS_NUM_THREADS"])
    builder = builder.config("spark.executorEnv.MKL_NUM_THREADS", os.environ["MKL_NUM_THREADS"])
    builder = builder.config("spark.executorEnv.NUMEXPR_NUM_THREADS", os.environ["NUMEXPR_NUM_THREADS"])
    builder = builder.config("spark.python.worker.faulthandler.enabled", "true")
    builder = builder.config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
    builder = builder.config("spark.python.worker.reuse", "true")
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    builder = builder.config("spark.ui.showConsoleProgress", "true")

    codebase_root = Path(__file__).resolve().parents[2]
    project_root = codebase_root.parent
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [str(codebase_root), str(project_root)]
    if existing_pythonpath.strip():
        pythonpath_parts.append(existing_pythonpath)
    executor_pythonpath = ":".join(pythonpath_parts)
    builder = builder.config("spark.executorEnv.PYTHONPATH", executor_pythonpath)

    for key, value in (spark_config or {}).items():
        builder = builder.config(str(key), str(value))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)
    return spark
