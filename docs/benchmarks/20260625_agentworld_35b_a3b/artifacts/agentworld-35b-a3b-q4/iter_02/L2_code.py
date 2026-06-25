from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        data = self.jobs[job_id]
        if 'retry_delays' not in data:
            data['retry_delays'] = []
        if 'retry_count' not in data:
            data['retry_count'] = 0
            
        backoff_delays = [1, 2, 4]
        
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    data['retry_delays'].append(backoff_delays[attempt])
                    data['retry_count'] = attempt + 1
                else:
                    data['retry_count'] = 3
                    
        return False