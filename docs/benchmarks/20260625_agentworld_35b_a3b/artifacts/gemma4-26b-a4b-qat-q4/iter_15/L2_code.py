from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self.jobs.get(job_id)
        if data is None:
            return False

        if 'delays' not in data:
            data['delays'] = []

        backoffs = [1, 2, 4]
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                processor(data)
                return True
            except Exception:
                # If we have retries left (we have already done 'attempt' failures)
                if attempt < max_attempts - 1:
                    delay = backoffs[attempt]
                    data['delays'].append(delay)
                else:
                    return False
        return False