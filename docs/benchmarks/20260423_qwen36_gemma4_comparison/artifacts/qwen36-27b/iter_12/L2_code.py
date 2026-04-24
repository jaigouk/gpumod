from typing import Callable, Any, Dict, List

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Any] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoff_delays: Dict[str, List[float]] = {}

    def add_job(self, job_id: str, data: Any) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.backoff_delays[job_id] = []

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job_data = self.jobs[job_id]
        self.retry_counts[job_id] = 0
        self.backoff_delays[job_id] = []

        backoff_times = [1, 2, 4]

        for attempt in range(4):
            try:
                processor(job_data)
                return True
            except Exception:
                if attempt < 3:
                    self.retry_counts[job_id] += 1
                    self.backoff_delays[job_id].append(backoff_times[attempt])

        return False