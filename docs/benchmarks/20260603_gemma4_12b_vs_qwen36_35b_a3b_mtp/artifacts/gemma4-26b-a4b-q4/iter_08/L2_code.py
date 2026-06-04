from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = {"data": data, "retries": 0}

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            max_retries = 3

            while job["retries"] <= max_retries:
                try:
                    processor(job["data"])
                    return True
                except Exception:
                    if job["retries"] == max_retries:
                        return False

                    # Calculate backoff: 1, 2, 4
                    # At retry 0 (first failure): wait 2^0 = 1
                    # At retry 1 (second failure): wait 2^1 = 2
                    # At retry 2 (third failure): wait 2^2 = 4
                    # Note: the prompt says "retry up to 3 times"
                    # If retry_count starts at 0.
                    # First failure: job['retries'] becomes 1.
                    # Wait 2^0? Or 1s, 2s, 4s.

                    # Let's refine the loop.
                    # Attempt 1. Fail. job['retries'] becomes 1. Wait 1s.
                    # Attempt 2. Fail. job['retries'] becomes 2. Wait 2s.
                    # Attempt 3. Fail. job['retries'] becomes 3. Wait 4s.
                    # Attempt 4. Fail. return False.
                    # Wait... "retry up to 3 times".
                    # Original attempt + 3 retries = 4 attempts.

                    # Let's use the prompt's logic: 1, 2, 4.
                    # Attempt 0 (original): fail.
                    # Retry 1: fail.
                    # Retry 2: fail.
                    # Retry 3: fail.
                    # total 4 attempts.
                    pass