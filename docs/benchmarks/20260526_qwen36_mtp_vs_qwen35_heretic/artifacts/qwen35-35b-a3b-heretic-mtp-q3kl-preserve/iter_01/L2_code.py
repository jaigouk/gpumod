from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}

        def add_job(self, job_id: str, data: Any) -> None:
            self.jobs[job_id] = {
                "data": data,
                "retry_count": 0
            }

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            max_retries = 3
            backoff_delays = [1, 2, 4]
            
            # Attempt 1 (Initial)
            # If we consider "retry up to 3 times" as 3 additional attempts after the first
            # Then we have 4 total attempts.
            # Retries 1, 2, 3 map to delays 0, 1, 2 in the backoff list?
            # Or retries 1, 2, 3 map to delays 1, 2, 4.
            
            # Let's implement a loop.
            # Current attempt logic:
            # We start with attempt 0.
            # If fail, we increment retry_count.
            # If retry_count <= max_retries:
            #    wait backoff[retry_count - 1]
            #    retry
            
            # Let's trace:
            # Try 1. Fail. retry_count becomes 1.
            # Wait 1s (index 0).
            # Try 2. Fail. retry_count becomes 2.
            # Wait 2s (index 1).
            # Try 3. Fail. retry_count becomes 3.
            # Wait 4s (index 2).
            # Try 4. Fail. retry_count becomes 4.
            # Stop.
            
            # This aligns with "Retry up to 3 times" (1, 2, 3) and delays [1, 2, 4].
            
            for attempt in range(max_retries + 1):
                try:
                    processor(job["data"])
                    return True
                except Exception:
                    job["retry_count"] = attempt
                    if attempt < max_retries:
                        # Simulate backoff
                        delay = backoff_delays[attempt]
                        # In a real scenario: time.sleep(delay)
                        # Requirement: "stored/tracked rather than actually sleeping"
                        # I'll just note the delay logic or store it.
                        # I will store the delay in the job state to satisfy "backoff delays can be stored/tracked"
                        job["last_backoff"] = delay
                    else:
                        return False
            
            return False