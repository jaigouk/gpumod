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

            data = self.jobs[job_id]
            max_retries = 3

            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    # If successful, reset retry count and return True
                    self.retry_counts[job_id] = 0
                    return True
                except Exception as e:
                    # Check if we can retry
                    if attempt < max_retries:
                        # Logically, we'd sleep here.
                        # Backoff: 1, 2, 4...
                        # delay = 2 ** attempt
                        self.retry_counts[job_id] += 1
                        continue # Proceed to next iteration
                    else:
                        return False
            return False