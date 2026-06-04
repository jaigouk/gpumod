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
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    # Tracking retry count and recording delay in data as instructed
                    data['retry_count'] = attempt
                    data['delay'] = 2 ** (attempt - 1)

                processor(data)
                return True
            except Exception:
                if attempt == max_attempts - 1:
                    return False
                continue

        return False