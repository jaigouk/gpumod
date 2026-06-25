from typing import Callable, Dict, List

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self.retry_counts: Dict[str, int] = {}
        self.retry_delays: Dict[str, List[float]] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.retry_delays[job_id] = []

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self.jobs.get(job_id)
        if data is None:
            return False

        for attempt in range(4):
            if attempt > 0:
                delay = 2 ** (attempt - 1)
                self.retry_delays[job_id].append(delay)
            try:
                processor(data)
                return True
            except Exception:
                self.retry_counts[job_id] = attempt
        return False