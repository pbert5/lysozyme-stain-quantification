from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple


def _resolve_processes_flag(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    posix_spawn = os.name == "posix" and sys.version_info >= (3, 13)
    return not posix_spawn


def _lazy_import_dask():
    """Import dask.distributed only when needed."""
    try:
        from dask.distributed import Client, LocalCluster  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - explicit failure mode
        raise RuntimeError(
            "dask.distributed is required to run the cluster helper.\n"
            "Install it in your environment, e.g. `pip install dask[distributed]`."
        ) from exc
    return Client, LocalCluster


def default_venv_python(root: Optional[Path] = None) -> Path:
    """
    Return the Python executable expected inside the repository's `.venv`.

    Parameters
    ----------
    root:
        Project root. Defaults to three directories above this file.

    Notes
    -----
    The returned path is not guaranteed to exist; callers can check with
    ``Path.exists()`` before using it in subprocess invocations.
    """
    if root is None:
        root = Path(__file__).resolve().parents[3]
    exe = "python.exe" if os.name == "nt" else "python"
    subdir = "Scripts" if os.name == "nt" else "bin"
    return root / ".venv" / subdir / exe


@contextmanager
def start_local_cluster(
    *,
    n_workers: Optional[int] = None,
    threads_per_worker: Optional[int] = None,
    memory_limit: Optional[str] = None,
    scheduler_port: Optional[int] = None,
    dashboard_address: str = ":8787",
    processes: Optional[bool] = None,
    silence_logs: bool | str | int = "error",
    set_as_default: bool = True,
) -> Iterator[Tuple["LocalCluster", "Client"]]:
    """
    Context manager that spins up a LocalCluster/Client pair and tears it down on exit.

    Examples
    --------
    >>> from image_ops_framework.helpers.cluster import start_local_cluster
    >>> with start_local_cluster(n_workers=2) as (cluster, client):
    ...     client.wait_for_workers(2)
    ...     print(cluster.dashboard_link)
    """
    Client, LocalCluster = _lazy_import_dask()

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        scheduler_port=scheduler_port,
        dashboard_address=dashboard_address,
        silence_logs=silence_logs,
        processes=_resolve_processes_flag(processes),
    )
    client = Client(cluster, set_as_default=set_as_default)
    try:
        yield cluster, client
    finally:
        client.close()
        cluster.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m image_ops_framework.helpers.cluster",
        description="Launch a reusable LocalCluster for interactive work.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example
            -------
                $ %(prog)s --n-workers 4 --dashboard-address 0.0.0.0:8787

            Forward the dashboard port (default 8787) to your desktop to inspect
            the Dask Web UI from VS Code.
            """
        ),
    )
    parser.add_argument("--n-workers", type=int, default=None, help="Number of worker processes.")
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="Thread pool size per worker process.",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default=None,
        help="Per-worker memory cap, e.g. '4GB'.",
    )
    parser.add_argument(
        "--scheduler-port",
        type=int,
        default=None,
        help="Fixed TCP port for the scheduler. Defaults to an ephemeral port.",
    )
    parser.add_argument(
        "--dashboard-address",
        type=str,
        default=":8787",
        help="Bind IP:port for the diagnostics dashboard. Use 0.0.0.0 for remote access.",
    )
    parser.add_argument(
        "--threads-only",
        action="store_true",
        help="Force a threaded cluster (no worker processes). Useful on Python 3.13+ when 'spawn' causes issues.",
    )
    parser.add_argument(
        "--no-set-default",
        action="store_true",
        help="Do not register the client as the global Dask default.",
    )
    return parser


def cli_main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    Client, LocalCluster = _lazy_import_dask()
    processes_flag = _resolve_processes_flag(False if args.threads_only else None)
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit,
        scheduler_port=args.scheduler_port,
        dashboard_address=args.dashboard_address,
        silence_logs="error",
        processes=processes_flag,
    )
    client = Client(cluster, set_as_default=not args.no_set_default)

    try:
        print("Local Dask cluster started.")
        print(f"  Scheduler address : {cluster.scheduler_address}")
        print(f"  Dashboard         : {cluster.dashboard_link or 'disabled'}")
        print("Press CTRL+C to stop.")

        target_workers = args.n_workers or len(cluster.workers)
        if target_workers:
            client.wait_for_workers(target_workers)

        mp.Event().wait()  # Block forever until interrupted.
    except KeyboardInterrupt:
        print("\nStopping cluster...")
    finally:
        client.close()
        cluster.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    mp.freeze_support()
    sys.exit(cli_main())
