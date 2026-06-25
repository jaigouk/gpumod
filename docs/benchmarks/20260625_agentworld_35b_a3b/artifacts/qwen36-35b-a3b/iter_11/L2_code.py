from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.retried_delays = []

    def add_job(self, job_id: str, data: dict) -> None:
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        delays = [1, 2, 4]
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    self.retried_delays.append(delays[attempt])
        return False