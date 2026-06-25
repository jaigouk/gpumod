from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> None:
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        attempts = 0
        max_attempts = 4

        while attempts < max_attempts:
            try:
                processor(data)
                return True
            except Exception:
                attempts += 1
                if attempts < max_attempts:
                    # Exponential backoff: 1s, 2s, 4s
                    delay = 2 ** (attempts - 1)
                    if 'delays' not in data:
                        data['delays'] = []
                    data['delays'].append(delay)
                else:
                    return False
        return False