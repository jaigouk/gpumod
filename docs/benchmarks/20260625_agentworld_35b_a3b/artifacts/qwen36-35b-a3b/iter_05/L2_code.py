from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self.jobs[job_id]
        delays = []
        for attempt in range(1, 5):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 4:
                    delay = 2 ** (attempt - 1)
                    delays.append(delay)
                    data['delays'] = delays
                    self.retry_counts[job_id] = attempt - 1
                else:
                    self.retry_counts[job_id] = 3
                    data['delays'] = delays
                    return False
        return False