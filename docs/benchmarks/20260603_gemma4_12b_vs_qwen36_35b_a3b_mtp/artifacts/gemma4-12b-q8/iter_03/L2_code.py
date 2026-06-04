from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]

        if "retry_count" not in data:
            data["retry_count"] = 0
        if "delays" not in data:
            data["delays"] = []

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    data["retry_count"] += 1
                    # Exponential backoff: 2^0=1, 2^1=2, 2^2=4
                    delay = 2 ** (data["retry_count"] - 1)
                    data["delays"].append(delay)
                else:
                    break

        return False