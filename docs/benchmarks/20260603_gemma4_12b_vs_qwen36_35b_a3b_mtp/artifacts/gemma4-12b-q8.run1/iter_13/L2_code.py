from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

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
                except Exception:
                    if attempt < max_retries:
                        # Simulate delay: 2**attempt (where attempt 0 is first failure)
                        # Wait, 2**0=1, 2**1=2, 2**2=4.
                        # So if attempt is 0, 1, 2.
                        # If attempt == max_retries, we've exhausted retries.
                        job["retries"] += 1
                        # print(f"Retrying job {job_id}, attempt {job['retries']}") # No prints requested
                        pass # Logic for backoff simulation would go here
                    else:
                        return False
            return False