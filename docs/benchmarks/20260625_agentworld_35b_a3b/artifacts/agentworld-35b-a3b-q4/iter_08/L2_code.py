from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        if job_id not in self._jobs:
            self._jobs[job_id] = {
                'data': data,
                'retry_delays': []
            }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job_record = self._jobs[job_id]
        data = job_record['data']
        retry_delays = job_record['retry_delays']
        
        delays = [1, 2, 4]
        attempt = 0
        max_attempts = 4
        
        while attempt < max_attempts:
            attempt += 1
            try:
                processor(data)
                return True
            except Exception:
                if attempt < max_attempts:
                    delay = delays[attempt - 1]
                    retry_delays.append(delay)
                else:
                    return False
        
        return False