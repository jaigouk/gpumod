from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        if job_id not in self._jobs:
            self._jobs[job_id] = {'data': data, 'delays': []}

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        
        data = job['data']
        delays = job['delays']
        
        backoff_delays = [1, 2, 4]
        
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delays.append(backoff_delays[attempt])
        
        return False