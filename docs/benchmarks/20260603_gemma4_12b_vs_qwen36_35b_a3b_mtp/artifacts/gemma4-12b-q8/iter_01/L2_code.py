from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = {
            "data": data,
            "retries": 0
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job_entry = self.jobs[job_id]
        data = job_entry["data"]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    job_entry["retries"] += 1
                    delay = 2 ** (job_entry["retries"] - 1)

                    if "delays" not in data:
                        data["delays"] = []
                    data["delays"].append(delay)
                else:
                    return False
        return False