from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data

    def process_job(self, job_id: str, Processor: Callable[[Dict[str, Any]]]) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]

        if "retry_count" not in data:
            data["retry_count"] = 0
        if "delays" not in data:
            data["delays"] = []

        for attempt in range(4):
            try:
                Processor(data)
                return True
            except Exception:
                if attempt < 3:
                    data["retry_count"] += 1
                    delay = 2 ** attempt
                    data["delays"].append(delay)
                else:
                    break

        return False