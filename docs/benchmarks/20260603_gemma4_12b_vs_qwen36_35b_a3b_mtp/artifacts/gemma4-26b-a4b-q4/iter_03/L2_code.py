from typing import Callable, Any, Dict

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
            backoff_delays = [1, 2, 4]
            
            while True:
                try:
                    processor(data)
                    return True
                except Exception:
                    current_retries = self.retry_counts.get(job_id, 0)
                    
                    if current_retries < len(backoff_delays):
                        delay = backoff_delays[current_retries]
                        # Simulating delay as requested
                        # In a real app: time.sleep(delay)
                        self.retry_counts[job_id] = current_retries + 1
                        # Logic for "tracking" could mean logging or just incrementing
                        continue 
                    else:
                        return False