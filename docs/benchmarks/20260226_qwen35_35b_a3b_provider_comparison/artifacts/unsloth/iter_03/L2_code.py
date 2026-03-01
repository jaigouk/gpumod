from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0,
            "status": "pending",
            "next_backoff": 0
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        backoffs = [1, 2, 4]
        max_retries = 3
        retry_count = 0

        while retry_count <= max_retries:
            try:
                processor(job["data"])
                job["retry_count"] = retry_count
                job["status"] = "completed"
                return True
            except Exception:
                retry_count += 1
                job["retry_count"] = retry_count
                
                if retry_count <= max_retries:
                    job["next_backoff"] = backoffs[retry_count - 1]
                    job["status"] = "retrying"
                else:
                    job["status"] = "failed"
                    return False