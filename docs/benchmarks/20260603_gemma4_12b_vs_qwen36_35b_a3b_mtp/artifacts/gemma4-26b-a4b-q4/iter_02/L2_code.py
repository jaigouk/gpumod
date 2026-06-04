from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}
            self.backoff_history: Dict[str, list] = {} # To track delays as requested

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0
            self.backoff_history[job_id] = []

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            max_retries = 3
            
            while self.retry_counts[job_id] <= max_retries:
                try:
                    processor(self.jobs[job_id])
                    return True
                except Exception as e:
                    if self.retry_counts[job_id] == max_retries:
                        return False
                    
                    # Calculate delay: 2^retry_count
                    # Retry 0 -> attempt 1 (fails) -> retry 1 happens
                    # delay = 2^0 = 1
                    # delay = 2^1 = 2
                    # delay = 2^2 = 4
                    delay = 2 ** self.retry_counts[job_id]
                    self.backoff_history[job_id].append(delay)
                    self.retry_counts[job_id] += 1
                    # In a real system, time.sleep(delay)
                    
            return False