from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self.jobs.get(job_id)
        if data is None:
            raise ValueError(f"Job {job_id} not found")
        
        if '_delays' not in data:
            data['_delays'] = []
            
        delays = [1, 2, 4]
        max_attempts = 4
        
        for attempt in range(1, max_attempts + 1):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < max_attempts:
                    delay = delays[attempt - 1]
                    data['_delays'].append(delay)
                else:
                    return False