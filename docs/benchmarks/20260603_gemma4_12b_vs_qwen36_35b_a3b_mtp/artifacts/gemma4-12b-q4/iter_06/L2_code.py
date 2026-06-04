from typing import Callable, Dict, Any
    import time

    class JobQueue:
        def __init__(self):
            self.jobs = {} # {job_id: {"data": ..., "retries": 0}}

        def add_job(self, job_id: str, data: dict):
            self.jobs[job_id] = {"data": data, "retries": 0}

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            max_retries = 3

            for attempt in range(max_retries + 1):
                try:
                    processor(job["data"])
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        job["retries"] += 1
                        # Exponential backoff: 1, 2, 4
                        delay = 2 ** attempt
                        print(f"Retry {job['retries']} for {job_id} after {delay}s delay due to: {e}")
                        # Actual sleep would be: time.sleep(delay)
                    else:
                        return False
            return False