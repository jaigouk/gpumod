from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job_data = self.jobs[job_id]
        if "retries" not in job_data:
            job_data["retries"] = 0

        for attempt in range(4):
            try:
                processor(job_data)
                return True
            except Exception:
                if attempt < 3:
                    # Exponential backoff delays: 2^0=1, 2^1=2, 2^2=4
                    delay = 2 ** attempt
                    job_data["retries"] = attempt + 1
                    job_data["last_delay"] = delay
                else:
                    return False
        return False