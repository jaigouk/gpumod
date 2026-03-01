from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0,
            "backoff_delays": []
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        
        # Prevent exceeding max retries if process_job is called multiple times
        if job["retry_count"] >= 3:
            return False

        max_retries = 3
        base_delay = 1

        # Attempt initial execution + up to 3 retries (4 attempts total)
        for attempt in range(max_retries + 1):
            try:
                processor(job["data"])
                return True
            except Exception:
                if attempt < max_retries:
                    job["retry_count"] += 1
                    delay = base_delay * (2 ** attempt)
                    job["backoff_delays"].append(delay)
                    # time.sleep(delay) # Simulated
                else:
                    # Max retries exhausted
                    return False
        
        return False