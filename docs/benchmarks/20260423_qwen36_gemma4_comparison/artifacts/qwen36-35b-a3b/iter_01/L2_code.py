from typing import Callable, Any, Dict

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Any) -> None:
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0,
            "backoff_delays": []
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        max_retries = 3
        base_delay = 1

        for attempt in range(max_retries + 1):
            try:
                processor(job["data"])
                return True
            except Exception:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    job["backoff_delays"].append(delay)
                    job["retry_count"] = attempt + 1
                else:
                    job["retry_count"] = max_retries
                    return False
        return False