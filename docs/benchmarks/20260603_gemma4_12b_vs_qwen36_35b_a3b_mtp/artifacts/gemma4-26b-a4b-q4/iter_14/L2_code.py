from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        max_attempts = 4

        for attempt in range(1, max_attempts + 1):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < max_attempts:
                    delay = 2 ** (attempt - 1)
                    if 'retries' not in data:
                        data['retries'] = []
                    data['retries'].append(delay)
                else:
                    return False
        return False