from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job_data = self.jobs[job_id]

        # Ensure tracking fields exist in the job data
        if 'retry_count' not in job_data:
            job_data['retry_count'] = 0
        if 'delays' not in job_data:
            job_data['delays'] = []

        # 4 total attempts (initial + 3 retries)
        for attempt in range(4):
            try:
                processor(job_data)
                return True
            except Exception:
                # If this was the last attempt, we return False
                if attempt == 3:
                    return False

                # Calculate exponential backoff delay
                # attempt 0 (1st fail) -> 1s
                # attempt 1 (2nd fail) -> 2s
                # attempt 2 (3rd fail) -> 4s
                delays = [1, 2, 4]
                delay = delays[attempt]

                # Record delay as data
                job_data['delays'].append(delay)

                # Track retry count per job
                job_data['retry_count'] = attempt + 1

        return False