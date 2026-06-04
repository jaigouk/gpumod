from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}
            self.backoff_history: Dict[str, list] = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0
            self.backoff_history[job_id] = []

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
                        self.backoff_history[job_id].append(delay)
                        self.retry_counts[job_id] += 1
                    else:
                        # All retries exhausted
                        self.retry_counts[job_id] += 1 # increment to show it's done
                        return False
            return False