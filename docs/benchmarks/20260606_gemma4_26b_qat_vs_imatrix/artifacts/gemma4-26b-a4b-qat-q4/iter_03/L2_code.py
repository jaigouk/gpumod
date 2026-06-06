from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        if 'delays' not in data:
            data['delays'] = []

        for attempt in range(1, 5):
            try:
                processor(data)
                return True
            except Exception:
                if attempt == 4:
                    return False

                # Exponential backoff: 1s, 2s, 4s
                delay = 2 ** (attempt - 1)
                data['delays'].append(delay)

        return False