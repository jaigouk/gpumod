from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Any] = {}
        self.retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Any):
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        if isinstance(data, dict) and 'delays' not in data:
            data['delays'] = []

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    self.retry_counts[job_id]] += 1
                    delay = 2 ** (self.retry_counts[job_id]] - 1)
                    if isinstance(data, dict) and 'delays' in data:
                        data['delays']].append(delay)
                else:
                    break

        return False