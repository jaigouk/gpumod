from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self.jobs.get(job_id)
        if data is None:
            raise KeyError(f"Job {job_id} not found")
        
        if '_delays' not in data:
            data['_delays'] = []
        
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    data['_delays'].append(delay)
        return False