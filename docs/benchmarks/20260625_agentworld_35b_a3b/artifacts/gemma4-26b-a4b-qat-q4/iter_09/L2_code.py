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
        # 4 total attempts: initial + 3 retries
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    # Exponential backoff: 2**0=1, 2**1=2, 2**2=4
                    data['delay'] = 2 ** attempt
                else:
                    return False
        return False