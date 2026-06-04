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
                except Exception as e:
                    job["retries"] += 1
                    if job["retries"] > max_retries:
                        return False
                    
                    # Backoff logic (simulated as per requirements)
                    # Delay = 2 ** (retry_count - 1)
                    # Or explicitly 1, 2, 4.
                    delay = 2 ** (job["retries"] - 1)
                    # print(f"Retry {job['retries']} after {delay}s delay due to: {e}")
                    # time.sleep(delay) # Optional
            return False