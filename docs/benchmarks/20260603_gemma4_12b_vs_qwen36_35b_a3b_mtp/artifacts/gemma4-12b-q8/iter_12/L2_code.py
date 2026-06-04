import time
    from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            max_retries = 3
            job_data = self.jobs.get(job_id)

            if job_data is None:
                return False

            for attempt in range(max_retries + 1):
                try:
                    processor(job_data)
                    self.retry_counts[job_id] = 0 # Reset on success
                    return True
                except Exception as e:
                    self.retry_counts[job_id] = attempt + 1
                    if attempt < max_retries:
                        # Exponential backoff: 1, 2, 4
                        delay = 2 ** attempt
                        time.sleep(delay) # Or simulate as requested
                    else:
                        return False
            return False