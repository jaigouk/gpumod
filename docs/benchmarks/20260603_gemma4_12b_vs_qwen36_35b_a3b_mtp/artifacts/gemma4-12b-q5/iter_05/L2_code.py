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
            data = self.jobs.get(job_id)
            
            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    return True
                except Exception:
                    if attempt < max_retries:
                        self.retry_counts[job_id] = attempt + 1
                        # Logic for backoff (1, 2, 4)
                        # Since we don't sleep, we just acknowledge the backoff exists
                        pass
                    else:
                        return False
            return False