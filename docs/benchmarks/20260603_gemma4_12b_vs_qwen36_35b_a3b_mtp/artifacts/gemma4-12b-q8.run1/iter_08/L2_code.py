from typing import Callable, Any

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.retry_counts = {}

        def add_job(self, job_id: str, data: dict):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable[[dict], Any]) -> bool:
            data = self.jobs.get(job_id)
            if not data:
                return False

            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    return True
                except Exception as e:
                    if attempt == max_retries:
                        return False
                    
                    self.retry_counts[job_id] += 1
                    delay = 2 ** (self.retry_counts[job_id] - 1)
                    print(f"Retry {self.retry_counts[job_id]} for {job_id} after {delay}s delay due to: {e}")
                    # Simulated sleep: time.sleep(delay)
            return False