from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self.jobs[job_id]
        max_attempts = 4

        for attempt in range(1, max_attempts + 1):
            try:
                processor(data)
                return True
            except Exception:
                if attempt == max_attempts:
                    return False

                # Calculate exponential backoff: 1s, 2s, 4s
                delay = 2 ** (attempt - 1)

                if 'retry_delays' not in data:
                    data['retry_delays'] = []
                data['retry_delays'].append(delay)

        return False