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
            # Note: Requirement says "retry up to 3 times".
            # This means Attempt 0 (original), then retries 1, 2, 3.

            for attempt in range(max_retries + 1):
                try:
                    # Call the processor with the job data
                    data = self.jobs[job_id]
                    processor(data)
                    # If successful, reset retry count and return True
                    self.retry_counts[job_id] = 0
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # Backoff calculation: 1, 2, 4...
                        # delay = 2 ** (self.retry_counts[job_id] - 1)
                        # But since we are skipping actual sleep:
                        pass
                    else:
                        # Exhausted retries
                        return False
            return False