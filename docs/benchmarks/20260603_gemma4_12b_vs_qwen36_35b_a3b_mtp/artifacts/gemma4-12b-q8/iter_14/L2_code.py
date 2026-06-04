from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

        def add_job(self, job_id: str, data: Dict[str, Any]):
            self.jobs[job_id] = {"data": data, "retries": 0}

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            data = job["data"]
            max_retries = 3

            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        job["retries"] += 1
                        # Exponential backoff: 1, 2, 4
                        backoff = 2 ** (job["retries"] - 1)
                        print(f"Retry {job['retries']} for {job_id} after {backoff}s delay (Error: {e})")
                    else:
                        return False
            return False