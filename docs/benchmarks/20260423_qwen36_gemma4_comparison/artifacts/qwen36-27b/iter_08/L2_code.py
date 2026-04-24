from typing import Callable, Any, Dict, List

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Any] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoff_delays: Dict[str, List[float]] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.backoff_delays[job_id] = []

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            raise KeyError(f"Job {job_id} not found")

        max_retries = 3
        retries = 0
        self.backoff_delays[job_id] = []

        while retries <= max_retries:
            try:
                processor(self.jobs[job_id])
                self.retry_counts[job_id] = retries
                return True
            except Exception:
                retries += 1
                if retries <= max_retries:
                    delay = 2 ** (retries - 1)
                    self.backoff_delays[job_id].append(delay)
                else:
                    self.retry_counts[job_id] = retries - 1
                    return False