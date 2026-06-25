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
        backoff_schedule = [1, 2, 4]
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < max_attempts - 1:
                    # Track retry count and record delay in the job's data
                    data['retry_count'] = data.get('retry_count', 0) + 1

                    if 'delays' not in data:
                        data['delays'] = []
                    data['delays'].append(backoff_schedule[attempt])
                else:
                    return False
        return False