from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Any] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoff_delays: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.backoff_delays[job_id] = 0

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        max_retries = 3
        backoff_schedule = [1, 2, 4]

        for attempt in range(max_retries + 1):
            try:
                processor(self.jobs[job_id])
                return True
            except Exception:
                self.retry_counts[job_id] += 1
                retry_count = self.retry_counts[job_id]

                if retry_count <= max_retries:
                    # Track backoff delay instead of sleeping
                    delay_index = retry_count - 1
                    if delay_index < len(backoff_schedule):
                        self.backoff_delays[job_id] = backoff_schedule[delay_index]
                    continue
                else:
                    return False
        
        return False