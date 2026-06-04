from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

        def add_job(self, job_id: str, data: dict):
            self.jobs[job_id] = {
                "data": data,
                "retry_count": 0
            }

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            max_retries = 3

            while True:
                try:
                    processor(job["data"])
                    return True
                except Exception:
                    if job["retry_count"] >= max_retries:
                        return False

                    job["retry_count"] += 1
                    # Simulation of backoff logic
                    # Delay would be 2 ** (job["retry_count"] - 1)
                    # Since we just need to track/simulate, no sleep needed
                    pass