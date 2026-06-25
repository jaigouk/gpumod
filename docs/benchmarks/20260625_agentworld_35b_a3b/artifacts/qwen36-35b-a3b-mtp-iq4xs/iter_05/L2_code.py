from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id, data):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]

        # Initialize delays list in the data dict if not present
        if 'delays' not in data:
            data['delays'] = []

        # 4 attempts total: 1 initial + 3 retries
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt == 3:
                    return False

                # Exponential backoff delays: 1s, 2s, 4s
                delay = [1, 2, 4][attempt]

                # Record delay as data
                data['delays'].append(delay)

        return False