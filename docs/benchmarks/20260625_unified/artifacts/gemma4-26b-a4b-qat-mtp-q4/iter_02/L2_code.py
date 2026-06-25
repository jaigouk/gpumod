from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.retry_counts = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        backoffs = [1, 2, 4]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = backoffs[attempt]
                    if 'retry_delays' not in data:
                        data['retry_delays'] = []
                    data['retry_delays'].append(delay)
                    self.retry_counts[job_id] = attempt + 1
                else:
                    return False
        return False