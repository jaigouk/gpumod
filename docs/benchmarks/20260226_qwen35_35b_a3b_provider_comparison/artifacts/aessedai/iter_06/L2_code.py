from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Any] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoffs: Dict[str, float] = {}

    def add_job(self, job_id: str, data: Any) -> None:
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        current_retry_count = self.retry_counts.get(job_id, 0)
        max_retries = 3

        while current_retry_count <= max_retries:
            try:
                processor(data)
                self.retry_counts[job_id] = 0
                self.backoffs[job_id] = 0
                return True
            except Exception:
                current_retry_count += 1
                self.retry_counts[job_id] = current_retry_count
                
                if current_retry_count > max_retries:
                    return False
                
                backoff = 2 ** (current_retry_count - 1)
                self.backoffs[job_id] = backoff
                # Backoff is tracked in self.backoffs, no actual sleep
        
        return False