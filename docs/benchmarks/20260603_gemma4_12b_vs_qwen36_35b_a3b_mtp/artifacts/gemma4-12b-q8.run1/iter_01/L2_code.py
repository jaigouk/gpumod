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
            backoff_schedule = [1, 2, 4]

            for attempt in range(max_retries + 1):
                try:
                    processor(self.jobs[job_id])
                    return True
                except Exception:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # Backoff happens here (simulated by just moving to next iteration)
                        # If actual sleep was needed: time.sleep(backoff_schedule[attempt])
                        continue
                    else:
                        return False
            return False