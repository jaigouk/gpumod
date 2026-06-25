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
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    data.setdefault('delays', []).append(delay)
                    data['retry_count'] = attempt + 1
                else:
                    data.setdefault('delays', []).append(4)
                    data['retry_count'] = 3
        return False