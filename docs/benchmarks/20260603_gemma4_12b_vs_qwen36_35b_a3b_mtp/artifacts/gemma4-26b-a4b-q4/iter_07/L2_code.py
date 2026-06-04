import time
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

            max_retries = 3
            data = self.jobs[job_id]

            while self.retry_counts[job_id] <= max_retries:
                try:
                    processor(data)
                    return True
                except Exception:
                    if self.retry_counts[job_id] == max_retries:
                        return False

                    # Calculate backoff: 1, 2, 4
                    delay = 2 ** self.retry_counts[job_id]
                    # Since requirements say "can be simulated" and "delays can be stored/tracked rather than actually sleeping"
                    # I will implement time.sleep for correctness but keep it simple.
                    # Or just print it.
                    # Actually, the prompt says "Write only the Python code".
                    time.sleep(delay)
                    self.retry_counts[job_id] += 1

            return False