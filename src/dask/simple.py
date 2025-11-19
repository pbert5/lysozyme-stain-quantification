from dask.distributed import Client, LocalCluster

# sticky_cluster.py
from dask.distributed import Client, LocalCluster

_client = None
_cluster = None

def get_client(
    n_workers=8, threads_per_worker=1, memory_limit="6GB",
    dashboard=":8787", local_dir="/tmp/dask", processes=True
) -> Client:
    global _client, _cluster
    if _client is None:
        _cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            dashboard_address=dashboard,
            local_directory=local_dir,
            processes=processes,
        )
        _client = Client(_cluster, set_as_default=True)
    return _client


if __name__ == "__main__":
    client = get_client()   # reuse this client anywhere in the same interpreter