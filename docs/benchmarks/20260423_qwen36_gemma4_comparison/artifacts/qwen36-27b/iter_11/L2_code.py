from typing import Callable, Any, Dict, List

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Any] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoff_delays: Dict[str, List[float]] = {}

    def add_job(self, job_id: str, data: Any) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.backoff_delays[job_id] = []

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            raise KeyError(f"Job {job_id} not found")

        job_data = self.jobs[job_id]
        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                processor(job_data)
                self.retry_counts[job_id] = 0
                self.backoff_delays[job_id] = []
                return True
            except Exception:
                retries += 1
                if retries <= max_retries:
                    delay = 2 ** (retries - 1)
                    self.backoff_delays[job_id].append(delay)
                    self.retry_counts[job_id] = retries
                    continue
        return False