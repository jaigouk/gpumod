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

        while True:
            try:
                processor(job["data"])
                return True
            except Exception:
                job["retry_count"] += 1
                if job["retry_count"] <= max_retries:
                    index = min(job["retry_count"] - 1, len(backoff_sequence) - 1)
                    job["backoff_delays"].append(backoff_sequence[index])
                else:
                    return False