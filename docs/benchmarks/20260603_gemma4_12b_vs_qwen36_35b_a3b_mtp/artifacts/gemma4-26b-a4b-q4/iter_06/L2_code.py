import time
    from typing import Callable, Any, Dict

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.retry_counts = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            max_retries = 3
            # The prompt says "retry up to 3 times". Usually means original + 3 retries = 4 attempts total?
            # Or 3 attempts total?
            # "retry up to 3 times" usually implies 3 *extra* attempts.
            # Let's look at backoff: 1s, 2s, 4s. That's 3 delays.
            # Attempt 1 (Fail) -> Wait 1s -> Attempt 2 (Fail) -> Wait 2s -> Attempt 3 (Fail) -> Wait 4s -> Attempt 4 (Fail).
            # Wait, the prompt says "retry up to 3 times".
            # If max_retries = 3:
            # Try 1 (fail)
            # Retry 1 (fail) after 1s
            # Retry 2 (fail) after 2s
            # Retry 3 (fail) after 4s
            # Total attempts = 4. Total retries = 3.
            # Let's stick to a loop approach.

            data = self.jobs[job_id]

            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    return True
                except Exception:
                    if attempt < max_retries:
                        # Calculate backoff: 2^attempt?
                        # 0: 1s, 1: 2s, 2: 4s
                        wait_time = 2**attempt
                        # Simulation: instead of time.sleep(wait_time), just log or track?
                        # The prompt says "can be simulated". I'll use time.sleep for a real implementation
                        # or just leave it as logic.
                        # Actually, the prompt says "The backoff delays can be stored/tracked rather than actually sleeping."
                        # This implies I don't *have* to sleep.
                        # I will implement the logic that tracks the count.
                        self.retry_counts[job_id] += 1
                        # time.sleep(wait_time)
                    else:
                        return False
            return False