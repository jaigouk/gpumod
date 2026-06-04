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
                data = self.jobs[job_id]

                while self.retry_counts[job_id] <= max_retries:
                    try:
                        processor(data)
                        return True
                    except Exception:
                        if self.retry_counts[job_id] == max_retries:
                            return False

                        # Calculate backoff: 2^retry_count
                        # retry 0 (fail) -> backoff 1s
                        # retry 1 (fail) -> backoff 2s
                        # retry 2 (fail) -> backoff 4s
                        # Note: retry_counts[job_id] starts at 0.
                        # First fail: count becomes 1. Delay for next attempt.
                        # We need to simulate the delay.
                        self.retry_counts[job_id] += 1
                        # Simulation of backoff logic (no actual sleep)
                        # In a real system, we'd schedule this.
                        # Here, we just follow the retry logic.
                        pass

                return False