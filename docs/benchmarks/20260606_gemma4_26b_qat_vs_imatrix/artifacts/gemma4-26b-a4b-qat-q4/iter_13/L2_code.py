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

        # Initialize tracking metadata if not present
        if 'retry_count' not in data:
            data['retry_count'] = 0
        if 'delays' not in data:
            data['delays'] = []

        backoff_schedule = [1, 2, 4]
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                processor(data)
                return True
            except Exception:
                # If we have retries left (max 3 retries for 4 total attempts)
                if attempt < max_attempts - 1:
                    delay = backoff_schedule[attempt]
                    data['retry_count'] += 1
                    data['delays'].append(delay)
                else:
                    return False
        return False