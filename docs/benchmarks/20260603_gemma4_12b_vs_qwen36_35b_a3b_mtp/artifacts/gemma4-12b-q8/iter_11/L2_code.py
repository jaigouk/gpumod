from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

        def add_job(self, job_id: str, data: Dict[str, Any]):
            self.jobs[job_id] = {"data": data, "retries": 0}

        def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]]]) -> bool:
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
                        # Backoff logic: 1s, 2s, 4s
                        # attempt 0 fail -> retry 1 (backoff 2^0 = 1)
                        # attempt 1 fail -> retry 2 (backoff 2^1 = 2)
                        # attempt 2 fail -> retry 3 (backoff 2^2 = 4)
                        # Since we only need to simulate/track, we can print or just continue.
                        pass
                    else:
                        return False
            return False