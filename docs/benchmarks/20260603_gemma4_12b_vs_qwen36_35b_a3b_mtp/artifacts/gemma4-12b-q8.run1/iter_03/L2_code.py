from typing import Callable, Any, Dict

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

        def add_job(self, job_id: str, data: Dict[str, Any]):
            self.jobs[job_id] = {
                "data": data,
                "retries": 0
            }

        def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]]) -> bool:
            if job_id not in self.jobs:
                return False

            max_retries = 3
            job = self.jobs[job_id]
            data = job["data"]

            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    return True
                except Exception:
                    if attempt < max_retries:
                        job["retries"] += 1
                        # Simulate backoff delay (e.g., print it or just acknowledge it)
                        # Delay = 2 ** (job["retries"] - 1)
                        pass
                    else:
                        return False
            return False