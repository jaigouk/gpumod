from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], None]) -> bool:
        data = self.jobs.get(job_id)
        if data is None:
            return False

        if "retry_count" not in data:
            data["retry_count"] = 0
        if "delays" not in data:
            data["delays"] = []

        for i in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if i < 3:
                    data["retry_count"] += 1
                    delay = 2 ** (data["retry_count"] - 1)
                    data["delays"].append(delay)
                else:
                    break
        return False