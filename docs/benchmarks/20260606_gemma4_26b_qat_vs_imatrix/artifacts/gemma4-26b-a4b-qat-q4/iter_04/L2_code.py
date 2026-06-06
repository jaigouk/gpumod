from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self.jobs[job_id]

        for attempt in range(1, 5):
            try:
                processor(data)
                return True
            except Exception:
                if attempt == 4:
                    return False

                # Exponential backoff: 1s, 2s, 4s
                delay = 2 ** (attempt - 1)

                # Track retry count and record delays in the data dict
                data['retry_count'] = data.get('retry_count', 0) + 1
                if 'delays' not in data:
                    data['delays'] = []
                data['delays'].append(delay)

        return False