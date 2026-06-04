from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

        def add_job(self, job_id: str, data: Dict[str, Any]):
            self.jobs[job_id] = {"data": data, "retries": 0}

        def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]]) -> bool:
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
                        # Backoff logic: 1, 2, 4 (2^(retries-1))
                        # Since job["retries"] is incremented,
                        # Attempt 0 (first fail) -> retries=1 -> delay 2^(1-1) = 1
                        # Attempt 1 (second fail) -> retries=2 -> delay 2^(2-1) = 2
                        # Attempt 2 (third fail) -> retries=3 -> delay 2^(3-1) = 4
                        pass # Simulation: delay = 2**(job["retries"] - 1)
                    else:
                        return False
            return False