from typing import Callable, Any

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.retry_counts = {}

        def add_job(self, job_id: str, data: dict):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            max_retries = 3

            for attempt in range(max_retries + 1):
                try:
                    # If it's a retry, record the count
                    if attempt > 0:
                        self.retry_counts[job_id] = attempt
                    
                    # Execute processor
                    # Assuming processor takes 'data' based on example
                    processor(data)
                    
                    # If successful, reset retry count and return True
                    self.retry_counts[job_id] = 0
                    return True
                except Exception as e:
                    # If we have retries left, we would "sleep" here
                    # But requirement says "backoff delays can be stored/tracked"
                    # "Use exponential backoff between retries: 1s, 2s, 4s"
                    # This implies we don't actually sleep.
                    pass
            
            # If we reach here, all retries exhausted
            self.retry_counts[job_id] = max_retries
            return False