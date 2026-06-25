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
        if 'retry_count' not in data:
            data['retry_count'] = 0

        backoff_schedule = [1, 2, 4]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    data['retry_count'] += 1
                    data['delays'].append(backoff_schedule[attempt])
                else:
                    return False
        return False