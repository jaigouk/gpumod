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
        for attempt in range(4):
            if attempt > 0:
                data['retry_delay'] = 2 ** (attempt - 1)

            try:
                processor(data)
                return True
            except Exception:
                if attempt == 3:
                    return False
        return False