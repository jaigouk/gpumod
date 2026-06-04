from typing import Callable, Dict, Any
    import time

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.retry_counts = {}

        def add_job(self, job_id: str, data: Dict[str, Any]):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            max_retries = 3
            job_data = self.jobs[job_id]

            for attempt in range(max_retries + 1):
                try:
                    processor(job_data)
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # Exponential backoff: 1, 2, 4
                        # delay = 2 ** attempt (if attempt starts at 0)
                        # Wait... Requirement says "1s, 2s, 4s" for 3 retries.
                        # attempt 0 (initial): fail -> retry 1 (delay 1)
                        # attempt 1: fail -> retry 2 (delay 2)
                        # attempt 2: fail -> retry 3 (delay 4)
                        # attempt 3: fail -> max reached.
                        delay = 2 ** attempt
                        # Simulate delay (no actual sleep required by prompt)
                        # print(f"Retrying {job_id} in {delay}s...")
                        pass # Simulated sleep
                    else:
                        return False
            return False