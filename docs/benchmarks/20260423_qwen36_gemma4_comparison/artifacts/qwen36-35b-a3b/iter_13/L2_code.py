from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Any] = {}
        self.retry_state: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data
        self.retry_state[job_id] = {"count": 0, "backoffs": [1, 2, 4]}

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        state = self.retry_state[job_id]
        max_retries = 3
        backoffs = state["backoffs"]

        for attempt in range(max_retries + 1):
            try:
                processor(self.jobs[job_id])
                return True
            except Exception:
                if attempt < max_retries:
                    state["count"] += 1
                    state["last_backoff"] = backoffs[attempt]
                else:
                    return False
        return False