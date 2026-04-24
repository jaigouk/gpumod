from typing import Callable, Any, Dict

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Any] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoff_delays: Dict[str, list] = {}

    def add_job(self, job_id: str, data: Any) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.backoff_delays[job_id] = []

    def process_job(self, job_id: str, processor: Callable) -> bool:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                processor(self.jobs[job_id])
                return True
            except Exception:
                self.retry_counts[job_id] = attempt + 1
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # 1, 2, 4
                    self.backoff_delays[job_id].append(delay)
        return False