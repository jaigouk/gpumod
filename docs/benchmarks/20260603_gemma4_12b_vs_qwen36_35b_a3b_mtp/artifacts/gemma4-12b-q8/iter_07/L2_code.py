from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.retry_counts = {}

        def add_job(self, job_id: str, data: Dict[str, Any]):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            max_retries = 3
            backoffs = [1, 2, 4]

            for attempt in range(max_retries + 1):
                try:
                    data = self.jobs[job_id]
                    processor(data)
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # Simulate sleep with backoffs[attempt]
                        # print(f"Retrying {job_id}...")
                        pass # Requirement says can be simulated
                    else:
                        return False
            return False