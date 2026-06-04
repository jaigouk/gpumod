from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]

        # Initialize tracking fields if they don't exist
        if "retry_count" not in data:
            data["retry_count"] = 0
        if "retry_delays" not in data:
            data["retry_delays"] = []

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                # Only record delays and retries if we have retries remaining
                if attempt < 3:
                    delay = 2 ** attempt
                    data["retry_delays"].append(delay)
                    data["retry_count"] += 1
                else:
                    break

        return False