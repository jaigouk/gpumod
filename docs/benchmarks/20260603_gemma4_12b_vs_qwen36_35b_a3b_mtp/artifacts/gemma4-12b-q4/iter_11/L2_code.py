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
            data = self.jobs[job_id]

            while True:
                try:
                    # Assuming the processor takes the data object
                    Processor(data)
                    self.retry_counts[job_id] = 0 # Reset on success
                    return True
                except Exception as e:
                    current_retries = self.retry_counts[job_id]
                    if current_retries >= max_retries:
                        return False

                    self.retry_counts[job_id] += 1
                    # Backoff calculation: 1, 2, 4...
                    # delay = 2 ** (self.retry_counts[job_id] - 1)
                    # Since we don't need to sleep, we just increment the count
                    # and loop again.
                    pass