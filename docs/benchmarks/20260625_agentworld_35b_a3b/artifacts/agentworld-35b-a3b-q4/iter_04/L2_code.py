from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict):
        if job_id not in self._jobs:
            self._jobs[job_id] = {'data': data, 'delays': []}

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self._jobs[job_id]
        data = job['data']
        delays = job['delays']
        
        backoff_delays = [1, 2, 4]
        attempt = 0
        
        while attempt < 4:
            attempt += 1
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 4:
                    delays.append(backoff_delays[attempt - 1])
                    
        return False