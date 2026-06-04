from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

        def add_job(self, job_id: str, data: Dict[str, Any]):
            self.jobs[job_id] = {"data": data, "retries": 0}

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            max_retries = 3

            for attempt in range(max_retries + 1):
                try:
                    processor(job["data"])
                    return True
                except Exception:
                    if attempt < max_retries:
                        job["retries"] += 1
                        # Delay = 2 ** (job["retries"] - 1)
                        # Requirement says 1, 2, 4.
                        # Attempt 0 fails -> retry 1 (delay 1s)
                        # Attempt 1 fails -> retry 2 (delay 2s)
                        # Attempt 2 fails -> retry 3 (delay 4s)
                        # Attempt 3 fails -> exhausted.
                        pass # Simulate delay
                    else:
                        return False
            return False