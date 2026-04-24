from typing import Callable, Dict, Any

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
            raise KeyError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        max_retries = 3
        backoff_delays = [1, 2, 4]

        while job["retry_count"] <= max_retries:
            try:
                processor(job["data"])
                return True
            except Exception:
                if job["retry_count"] < max_retries:
                    job["backoff_delays"].append(backoff_delays[job["retry_count"]])
                    job["retry_count"] += 1
                else:
                    return False