from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0,
            "delays": []
        }

    def process_job(self, job_id: str, processor: Callable[[dict], Any]) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        data = job["data"]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    job["retry_count"] += 1
                    delay = 2 ** (job["retry_count"] - 1)
                    job["delays"].append(delay)
                else:
                    break
        return False