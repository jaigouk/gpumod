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
        backoffs = [1, 2, 4]
        attempts = 0
        max_attempts = 4

        while attempts < max_attempts:
            try:
                processor(data)
                return True
            except Exception:
                attempts += 1
                if attempts < max_attempts:
                    delay = backoffs[attempts - 1]
                    if 'delays' not in data:
                        data['delays'] = []
                    data['delays'].append(delay)
                    data['retry_count'] = attempts
                else:
                    return False
        return False