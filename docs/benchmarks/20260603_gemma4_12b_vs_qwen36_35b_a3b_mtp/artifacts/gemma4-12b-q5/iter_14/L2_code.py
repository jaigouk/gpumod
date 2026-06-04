from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

        def add_job(self, job_id: str, data: dict):
            self.jobs[job_id] = {"data": data, "retries": 0}

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job_info = self.jobs[job_id]
            max_retries = 3

            for attempt in range(max_retries + 1):
                try:
                    processor(job_info["data"]))
                    return True
                except Exception as e:
                    job_info["retries"] += 1
                    if job_info["retries"] > max_retries:
                        return False
                    # Simulation of backoff:
                    # delay = 2 ** (job_info["retries"] - 1)
                    # print(f"Retrying after {delay}s...")
            return False