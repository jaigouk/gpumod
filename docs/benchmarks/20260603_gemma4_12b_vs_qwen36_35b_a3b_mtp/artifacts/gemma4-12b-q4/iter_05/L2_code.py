from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = {
            "data": data,
            "retries": 0
        }

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job_record = self.jobs[job_id]
        data = job_record["data"]

        for attempt in range(4):
            try:
                Processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    job_record["retries"] += 1
                    job_record["last_delay"] = delay
                else:
                    return False
        return False