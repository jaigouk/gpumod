from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        delays = []
        max_attempts = 4

        for attempt in range(max_attempts):
            if attempt > 0:
                delay = 2 ** (attempt - 1)
                delays.append(delay)
                data['delays'] = delays

            try:
                processor(data)
                return True
            except Exception:
                if attempt == max_attempts - 1:
                    return False
        return False