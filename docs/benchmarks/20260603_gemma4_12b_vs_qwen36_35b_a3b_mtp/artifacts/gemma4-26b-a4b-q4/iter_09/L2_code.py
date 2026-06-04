from typing import Callable, Any, Dict

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}
            self.backoff_logs: Dict[str, list] = {} # To track the "simulated" delays

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0
            self.backoff_logs[job_id] = []

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            max_retries = 3

            while self.retry_counts[job_id] <= max_retries:
                try:
                    processor(data)
                    return True
                except Exception:
                    if self.retry_counts[job_id] < max_retries:
                        # Calculate backoff: 1, 2, 4
                        delay = 2 ** self.retry_counts[job_id]
                        self.backoff_logs[job_id].append(delay)
                        self.retry_counts[job_id] += 1
                        # In a real system, we'd sleep or schedule.
                        # Here we simulate by just continuing the loop.
                    else:
                        self.retry_counts[job_id] += 1 # ensure count is up
                        return False
            return False