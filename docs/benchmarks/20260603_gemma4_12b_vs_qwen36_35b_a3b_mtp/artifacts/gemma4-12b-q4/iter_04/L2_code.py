from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = {
            "data": data,
            "retries": 0,
            "delays": []
        }

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]]]) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        backoffs = [1, 2, 4]

        for attempt in range(4):
            try:
                processor(job["data"])
                return True
            except Exception:
                if attempt < 3:
                    job["retries"] += 1
                    delay = backoffs[job["retries"] - 1]
                    job["delays"] .append(delay)
                else:
                    return False
        return False