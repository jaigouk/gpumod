from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = {"data": data, "retries": 0}

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            max_retries = 3
            
            while True:
                try:
                    processor(job["data"])
                    return True
                except Exception:
                    if job["retries"] < max_retries:
                        # Exponential backoff: 2^0=1, 2^1=2, 2^2=4
                        # We don't sleep, just simulate/track
                        # delay = 2 ** job["retries"]
                        job["retries"] += 1
                    else:
                        return False