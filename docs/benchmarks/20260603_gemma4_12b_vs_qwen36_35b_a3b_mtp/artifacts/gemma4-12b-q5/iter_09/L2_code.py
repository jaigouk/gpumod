from typing import Callable, Dict, Any

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
                        processor(data)
                        return True
                    except Exception as e:
                        self.retry_counts[job_id] += 1
                        if self.retry_counts[job_id] > max_retries:
                            return False
                        # Backoff logic
                        delay = 2 ** (self.retry_counts[job_id] - 1)
                        # print(f"Retrying job {job_id} in {delay}s...")
                return False