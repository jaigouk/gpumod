from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.retry_counts = {}
        self.delays_recorded = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.delays_recorded[job_id] = []

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        max_attempts = 4
        backoff_delays = [1, 2, 4]

        for attempt in range(max_attempts):
            try:
                processor(data)
                return True
            except Exception:
                if attempt == max_attempts - 1:
                    return False
                else:
                    delay = backoff_delays[attempt]
                    self.retry_counts[job_id] += 1
                    self.delays_recorded[job_id].append(delay)
                    # Do not actually sleep
                    # time.sleep(delay)

        return False