from typing import Callable, Dict, Any, List

class Job:
    def __init__(self, job_id: str, data: Dict[str, Any]):
        self.job_id = job_id
        self.data = data
        self.delays: List[float] = []

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = Job(job_id, data)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        data = job.data
        attempts = 0
        max_attempts = 4
        # Exponential backoff delays: 1s, 2s, 4s
        backoff_delays = [1, 2, 4]

        while attempts < max_attempts:
            try:
                processor(data)
                return True
            except Exception:
                attempts += 1
                if attempts < max_attempts:
                    # Record the delay as data
                    job.delays.append(backoff_delays[attempts - 1])
                    # Do not actually sleep

        return False