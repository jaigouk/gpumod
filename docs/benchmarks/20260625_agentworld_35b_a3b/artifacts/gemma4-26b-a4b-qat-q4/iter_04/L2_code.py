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
        if 'delays' not in data:
            data['delays'] = []

        backoff_schedule = [1, 2, 4]

        # Total of 4 attempts (initial + 3 retries)
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                # If we have more attempts remaining, record the next delay
                if attempt < 3:
                    data['delays'].append(backoff_schedule[attempt])
                else:
                    # All 4 attempts have been exhausted
                    return False
        return False