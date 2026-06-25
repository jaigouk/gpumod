from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = {'data': data, 'retries': 0}

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False

        data = job['data']
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    job['retries'] += 1
                    data['delay'] = 2 ** attempt
                else:
                    return False
        return False