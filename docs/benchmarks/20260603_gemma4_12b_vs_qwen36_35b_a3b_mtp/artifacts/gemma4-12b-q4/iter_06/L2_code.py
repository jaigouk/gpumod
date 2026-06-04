from typing import Callable, Dict, List

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self.retries: Dict[str, int] = {}
        self.delays: Dict[str, List[int]] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data
        self.retries[job_id] = 0
        self.delays[job_id]] = []

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                Processor(data)
                return True
            except Exception:
                if attempt < max_attempts - 1:
                    # Backoff sequence: 2^0=1, 2^1=2, 2^2=4
                    delay = 2 ** attempt
                    self.retries[job_id] += 1
                    self.delays[job_id]].append(delay)
                else:
                    break

        return False