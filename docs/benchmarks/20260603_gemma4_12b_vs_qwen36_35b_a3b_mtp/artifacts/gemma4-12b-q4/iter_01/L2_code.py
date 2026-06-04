from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, Processor: Callable[[dict], Any]) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        backoff_delays = [1, 2, 4]

        for attempt_idx in range(4):
            try:
                Processor(data)
                return True
            except Exception:
                if attempt_idx < 3:
                    data["retry_count"] = attempt_idx + 1
                    data["last_delay"] = backoff_delays[attempt_idx]
                else:
                    return False
        return False