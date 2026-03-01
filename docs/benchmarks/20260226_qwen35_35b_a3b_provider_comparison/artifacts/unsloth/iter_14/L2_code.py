from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        self.retry_counts[job_id] = 0
        max_retries = 3
        backoff_sequence = [1, 2, 4]
        current_retry = 0

        while current_retry <= max_retries:
            try:
                processor(self.jobs[job_id])
                return True
            except Exception:
                current_retry += 1
                self.retry_counts[job_id] = current_retry

                if current_retry <= max_retries:
                    delay = backoff_sequence[current_retry - 1]
                    # Simulate backoff (e.g., store delay or skip sleep)
                else:
                    return False
        return False