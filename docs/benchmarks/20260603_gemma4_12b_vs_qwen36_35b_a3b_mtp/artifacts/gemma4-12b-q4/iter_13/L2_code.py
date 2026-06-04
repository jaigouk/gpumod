from typing import Callable, Any, Dict

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.retry_counts = {}
            self.backoff_schedule = [1, 2, 4]

        def add_job(self, job_id: str, data: Dict[str, Any]):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]]]) -> bool:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            max_retries = 3

            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    self.retry_counts[job_id] = 0
                    return True
                except Exception as e:
                    self.retry_counts[job_id] += 1
                    if self.retry_counts[job_id] > max_retries:
                        break
                    # Simulate backoff logic
                    # delay = self.backoff_schedule[self.retry_counts[job_id] - 1]
                    # print(f"Retrying {job_id} in {delay}s...")

            self.retry_counts[job_id] = 0 # Reset or keep? Requirements say "Track retry count", usually implies clearing on success/failure or keeping it for the next time.
            return False