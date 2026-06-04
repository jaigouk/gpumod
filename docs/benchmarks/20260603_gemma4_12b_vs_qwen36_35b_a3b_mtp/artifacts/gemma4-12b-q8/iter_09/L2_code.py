from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        if "retry_history" not in data:
            data["retry_history"] = []
        if "retry_count" not in data:
            data["retry_count"] = 0
        self.jobs[job_id] = data

    def process_job(self, job_id: str, Processor: Callable[[Dict[str, Any]], Any]) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]]
        for attempt in range(4):
            try:
                Processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    data["retry_history"].append(delay))
                    data["retry_count"] += 1
                else:
                    break
        return False