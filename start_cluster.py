"""Optional persistent local CPU cluster. Stop with Ctrl-C."""
import argparse
import json
import time
from distributed import Client, LocalCluster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--port", type=int, default=8786)
    args = parser.parse_args()
    if args.workers < 1 or args.threads_per_worker < 1:
        parser.error("workers and threads must be positive")
    with LocalCluster(n_workers=args.workers, threads_per_worker=args.threads_per_worker,
                      processes=True, host="127.0.0.1", scheduler_port=args.port,
                      dashboard_address=None) as cluster:
        with Client(cluster) as client:
            print(json.dumps({"scheduler_address": cluster.scheduler_address,
                              "workers": list(client.scheduler_info()["workers"])}, indent=2), flush=True)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
