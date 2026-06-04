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
        backoffs = [1, 2, 4]
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < max_attempts - 1:
                    data['retry_count'] = attempt + 1
                    data['delay'] = backoffs[attempt]
                else:
                    return False
        return False