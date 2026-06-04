from typing import Callable, Dict

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self.retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: dict) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]

        while self.retry_counts[job_id] < 4:
            try:
                Processor(data)
                return True
            except Exception:
                if self.retry_counts[job_id] < 3:
                    delay = 2 ** self.retry_counts[job_id]
                    if 'delays' not in data:
                        data['delays'] = []
                    data['delays'].append(delay)
                    self.retry_counts[job_id]] += 1
                else:
                    self.retry_counts[job_id]] += 1
                    break
        return False