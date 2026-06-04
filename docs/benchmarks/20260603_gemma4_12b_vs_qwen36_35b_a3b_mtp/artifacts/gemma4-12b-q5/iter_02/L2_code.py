from typing import Callable, Any, Dict

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            data = self.jobs.get(job_id)
            if data is None:
                return False

            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    return True
                except Exception:
                    if attempt == max_retries:
                        return False
                    
                    # Exponential backoff: 1, 2, 4
                    # attempt 0 (fail) -> delay 2^0 = 1
                    # attempt 1 (fail) -> delay 2^1 = 2
                    # attempt 2 (fail) -> delay 2^2 = 4
                    # attempt 3 (fail) -> loop ends
                    
                    self.retry_counts[job_id] += 1
                    delay = 2 ** (self.retry_counts[job_id] - 1)
                    # Simulating sleep as per instructions
                    # print(f"Retrying {job_id} in {delay}s...")
            return False