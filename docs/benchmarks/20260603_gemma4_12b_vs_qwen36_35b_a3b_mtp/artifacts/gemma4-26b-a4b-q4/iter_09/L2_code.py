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
        # Delays for attempts: attempt 0 (none), attempt 1 (1s), attempt 2 (2s), attempt 3 (4s)
        delays = [0, 1, 2, 4]

        for attempt in range(4):
            if attempt > 0:
                data['delay'] = delays[attempt]

            try:
                processor(data)
                return True
            except Exception:
                if attempt == 3:
                    return False

        return False