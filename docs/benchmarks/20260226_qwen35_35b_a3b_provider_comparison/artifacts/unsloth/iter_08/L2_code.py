from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0,
            "backoff_delays": []
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        max_retries = 3
        backoff_sequence = [1, 2, 4]
        current_retry = 0

        while current_retry <= max_retries:
            try:
                processor(job["data"])
                job["retry_count"] = current_retry
                return True
            except Exception:
                job["retry_count"] = current_retry
                if current_retry < max_retries:
                    job["backoff_delays"].append(backoff_sequence[current_retry])
                current_retry += 1

        return False