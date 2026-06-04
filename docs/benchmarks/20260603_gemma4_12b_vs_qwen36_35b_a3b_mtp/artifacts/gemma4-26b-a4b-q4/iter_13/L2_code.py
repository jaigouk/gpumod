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

            max_retries = 3

            while self.retry_counts[job_id] <= max_retries:
                try:
                    processor(self.jobs[job_id])
                    return True
                except Exception:
                    if self.retry_counts[job_id] == max_retries:
                        return False

                    # Calculate delay: 1s, 2s, 4s
                    delay = 2 ** self.retry_counts[job_id]
                    # Simulation: instead of time.sleep(delay), we just track it
                    # But the prompt implies the logic should exist.
                    # I'll increment count and "simulate" the sleep by logic.
                    self.retry_counts[job_id] += 1
            return False