from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Any] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        delays = [1, 2, 4]

        for attempt in range(4):
            try:
                if attempt > 0:
                    data['retry_count'] = attempt
                    data['delay'] = delays[attempt - 1]

                processor(data)
                return True
            except Exception:
                if attempt == 3:
                    return False
        return False