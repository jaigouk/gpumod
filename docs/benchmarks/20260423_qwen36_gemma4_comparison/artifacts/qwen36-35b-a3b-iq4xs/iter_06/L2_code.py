from typing import Callable, Any, Dict, List

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = {
            "data": data,
            "status": "pending",
            "retry_count": 0,
            "backoff_delays": []
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        max_retries = 3

        for attempt in range(max_retries + 1):
            try:
                processor(job["data"])
                job["status"] = "completed"
                return True
            except Exception:
                if attempt < max_retries:
                    job["retry_count"] += 1
                    delay = 2 ** attempt
                    job["backoff_delays"].append(delay)
                    # time.sleep(delay)  # Simulated
                else:
                    job["status"] = "failed"
                    return False
        return False