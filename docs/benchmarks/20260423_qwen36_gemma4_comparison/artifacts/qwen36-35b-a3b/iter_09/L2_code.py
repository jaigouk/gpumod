from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._backoff_delays = [1, 2, 4]

    def add_job(self, job_id: str, data: Any) -> None:
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0,
            "status": "pending",
            "backoff": None
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        max_retries = 3

        for attempt in range(max_retries + 1):
            try:
                processor(job["data"])
                job["status"] = "completed"
                job["backoff"] = None
                return True
            except Exception:
                job["retry_count"] += 1
                if job["retry_count"] > max_retries:
                    job["status"] = "failed"
                    return False
                job["backoff"] = self._backoff_delays[job["retry_count"] - 1]
        return False