from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        backoff_delays = [1, 2, 4]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                # If this was the 4th attempt (index 3), fail.
                if attempt == 3:
                    return False

                # Record delay and increment retry count
                delay = backoff_delays[attempt]
                if 'delays' not in data:
                    data['delays'] = []
                data['delays'].append(delay)
                data['retry_count'] = data.get('retry_count', 0) + 1

        return False