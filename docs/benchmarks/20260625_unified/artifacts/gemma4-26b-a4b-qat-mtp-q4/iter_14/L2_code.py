from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self.jobs[job_id]
        backoffs = [1, 2, 4]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = backoffs[attempt]
                    data['retry_count'] = data.get('retry_count', 0) + 1
                    if 'delays' not in data:
                        data['delays'] = []
                    data['delays'].append(delay)
                else:
                    return False
        return False