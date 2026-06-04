import time
    from typing import Callable, Dict, Any

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
            max_retries = 3

            while self.retry_counts[job_id] <= max_retries:
                try:
                    processor(data)
                    return True
                except Exception:
                    if self.retry_counts[job_id] == max_retries:
                        return False

                    # Exponential backoff: 1, 2, 4...
                    # retry_counts[job_id] starts at 0.
                    # 1st fail: count becomes 1, delay 2^(1-1) = 1
                    # 2nd fail: count becomes 2, delay 2^(2-1) = 2
                    # 3rd fail: count becomes 3, delay 2^(3-1) = 4
                    # But wait, let's check loop condition.

                    # If I use a simple loop:
                    # attempt 0 -> fail -> count=1, delay=1
                    # attempt 1 -> fail -> count=2, delay=2
                    # attempt 2 -> fail -> count=3, delay=4
                    # attempt 3 -> fail -> return False