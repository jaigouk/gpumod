from typing import Callable, Dict, Any, List

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Any] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoff_delays: Dict[str, List[float]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.backoff_delays[job_id] = []

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            raise KeyError(f"Job '{job_id}' not found")

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                processor(self.jobs[job_id])
                return True
            except Exception:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    self.backoff_delays[job_id].append(delay)
                    self.retry_counts[job_id] += 1
                else:
                    return False