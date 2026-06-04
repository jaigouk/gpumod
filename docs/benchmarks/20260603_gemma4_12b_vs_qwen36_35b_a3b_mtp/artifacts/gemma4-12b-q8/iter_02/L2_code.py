from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        if "delays" not in data:
            data["delays"] = []
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job_entry = self.jobs[job_id]
        data = job_entry["data"]

        for _ in range(4):
            try:
                processor(data)
                return True
            except Exception:
                job_entry["retry_count"] += 1
                if job_entry["retry_count"] < 4:
                    delay = 2 ** (job_entry["retry_count"] - 1)
                    data["delays"].append(delay)
                else:
                    break

        return False