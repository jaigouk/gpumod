from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict) -> None:
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job_data = self.jobs[job_id]
        delays = []

        for attempt in range(4):
            try:
                processor(job_data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    delays.append(delay)
                    job_data['delays'] = delays
                    job_data['retry_count'] = attempt + 1

        return False