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
        max_retries = 3
        base_delay = 1.0
        delay = base_delay

        for attempt in range(max_retries + 1):
            try:
                processor(self.jobs[job_id])
                self.retry_counts[job_id] = attempt
                return True
            except Exception:
                if attempt < max_retries:
                    self.backoff_delays[job_id].append(delay)
                    delay *= 2
                else:
                    self.retry_counts[job_id] = max_retries
                    return False