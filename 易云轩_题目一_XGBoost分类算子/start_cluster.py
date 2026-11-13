"""Optional persistent local CPU cluster. Stop with Ctrl-C."""
import argparse
import time
from distributed import Client, LocalCluster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-workers", "--workers", dest="workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--port", type=int, default=8786)
    args = parser.parse_args()
    if args.workers < 1 or args.threads_per_worker < 1:
        parser.error("workers and threads must be positive")
    with LocalCluster(n_workers=args.workers, threads_per_worker=args.threads_per_worker,
                      processes=True, host="127.0.0.1", scheduler_port=args.port,
                      dashboard_address=None) as cluster:
        with Client(cluster) as client:
            client.wait_for_workers(args.workers)
            ready = len(client.scheduler_info()["workers"])
            print(f"Scheduler: {cluster.scheduler_address}\n"
                  f"Workers ready: {ready}/{args.workers} | Threads per worker: {args.threads_per_worker}\n"
                  "Stop with Ctrl-C.", flush=True)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
