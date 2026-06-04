from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

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
                    if self.retry_counts[job_id] == max_retries:
                        return False

                    # Increment retry count
                    self.retry_counts[job_id] += 1
                    # Logic for exponential backoff (1s, 2s, 4s)
                    # backoff = 2 ** (self.retry_counts[job_id] - 1)
                    # Since the prompt says "can be simulated/tracked rather than actually sleeping"
                    # I will just include a comment or a placeholder for the delay.
                    pass

            return False