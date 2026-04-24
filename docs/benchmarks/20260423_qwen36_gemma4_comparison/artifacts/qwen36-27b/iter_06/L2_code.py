from typing import Callable, Any, Dict

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Any] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoff_log: Dict[str, list] = {}

    def add_job(self, job_id: str, data: Any):
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.backoff_log[job_id] = []

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            raise KeyError(f"Job {job_id} not found")

        max_retries = 3
        backoff_schedule = [1, 2, 4]

        self.retry_counts[job_id] = 0
        self.backoff_log[job_id] = []

        for attempt in range(max_retries + 1):
            try:
                processor(self.jobs[job_id])
                return True
            except Exception:
                if attempt < max_retries:
                    self.retry_counts[job_id] += 1
                    self.backoff_log[job_id].append(backoff_schedule[attempt])
                else:
                    return False