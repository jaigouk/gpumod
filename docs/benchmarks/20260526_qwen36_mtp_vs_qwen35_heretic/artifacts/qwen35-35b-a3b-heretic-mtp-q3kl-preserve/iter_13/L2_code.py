import time
    import random

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.retry_counts = {}

        def add_job(self, job_id: str, data: dict):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: callable) -> bool:
            if job_id not in self.jobs:
                return False
            
            max_retries = 3
            backoff_delays = [1, 2, 4] # seconds
            
            retry_count = self.retry_counts.get(job_id, 0)
            
            # Attempt 1
            try:
                processor(self.jobs[job_id])
                return True
            except Exception:
                # Retry logic
                attempts_made = 1
                while attempts_made < max_retries + 1: # Total attempts = max_retries + 1
                    # Wait before retry if applicable
                    wait_idx = attempts_made - 1
                    if wait_idx < len(backoff_delays):
                        # Simulate sleep
                        time.sleep(backoff_delays[wait_idx])
                    
                    try:
                        processor(self.jobs[job_id])
                        return True
                    except Exception:
                        attempts_made += 1
                        retry_count = attempts_made - 1
                        self.retry_counts[job_id] = retry_count
                        
                return False