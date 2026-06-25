from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = {'data': data, 'delays': []}

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            raise KeyError(f"Job {job_id} not found")
        
        job_record = self.jobs[job_id]
        data = job_record['data']
        delays = job_record['delays']
        
        backoff_delays = [1, 2, 4]
        
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delays.append(backoff_delays[attempt])
        
        return False