from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                processor(data)
                return True
            except Exception:
                # If we haven't exhausted all 4 attempts, record the delay for the next one
                if attempt < max_attempts - 1:
                    # Exponential backoff: 2^0=1, 2^1=2, 2^2=4
                    data['delay'] = 2 ** attempt
                else:
                    # All 4 attempts failed
                    return False
        return False