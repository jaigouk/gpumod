from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self._delays = {}
        self._retry_counts = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        self._retry_counts[job_id] = 0
        self._delays[job_id] = []

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    self._delays[job_id].append(delay)
                    self._retry_counts[job_id] += 1
        return False