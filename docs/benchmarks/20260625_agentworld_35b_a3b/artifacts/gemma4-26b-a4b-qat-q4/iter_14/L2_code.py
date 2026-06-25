from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self.jobs.get(job_id)
        if data is None:
            return False

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    # Exponential backoff: 1s, 2s, 4s
                    data['delay'] = 2 ** attempt
                else:
                    return False
        return False