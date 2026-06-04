from typing import Callable, Any, Dict

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0,
            "delays": []
        }

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]]) -> bool:
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
                    # Exponential backoff: 1s, 2s, 4s
                    delay = 2 ** attempt
                    job_entry["retry_count"] += 1
                    job_entry["delays"].append(delay)
                else:
                    return False
        return False