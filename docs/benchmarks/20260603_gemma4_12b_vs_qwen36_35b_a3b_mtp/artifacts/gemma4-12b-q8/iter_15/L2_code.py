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

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    # Exponential backoff: 2^0=1, 2^1=2, 2^2=4
                    delay = 2 ** attempt
                    data["retry_count"] = attempt + 1
                    data["last_delay"] = delay
                else:
                    return False
        return False