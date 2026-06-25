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
        if 'retries' not in data:
            data['retries'] = 0
        if 'delays' not in data:
            data['delays'] = []

        max_attempts = 4
        for attempt_idx in range(max_attempts):
            try:
                processor(data)
                return True
            except Exception:
                # If we have more attempts remaining (initial + 3 retries = 4 total)
                if attempt_idx < max_attempts - 1:
                    delay = 2 ** attempt_idx  # 1, 2, 4
                    data['retries'] += 1
                    data['delays'].append(delay)
                else:
                    return False
        return False