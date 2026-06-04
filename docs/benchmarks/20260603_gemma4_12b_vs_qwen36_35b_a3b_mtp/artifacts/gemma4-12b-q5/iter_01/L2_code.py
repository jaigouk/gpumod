from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    # Exponential backoff: 2^0=1, 2^1=2, 2^2=4
                    delay = 2 ** attempt
                    data['next_retry_delay'] = delay
                    self.retry_counts[job_id] += 1
                else:
                    break

        return False