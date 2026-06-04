from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = {
            "data": data,
            "retries": 0
        }

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job_entry = self.jobs[job_id]
        data = job_entry["data"]

        if "delays" not in data:
            data["delays"] = []

        for attempt in range(4):
            try:
                Processor(data)
                return True
            except Exception:
                if attempt < 3:
                    job_entry["retries"] += 1
                    delay = 2 ** (job_entry["retries"] - 1)
                    data["delays"].append(delay)
                else:
                    return False
        return False