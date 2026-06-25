from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict):
        if job_id not in self._jobs:
            self._jobs[job_id] = {'data': data, 'delays': []}
        else:
            self._jobs[job_id]['data'] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        
        data = job['data']
        delays = job['delays']
        
        backoff_delays = [1, 2, 4]
        
        for attempt in range(1, 5):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 4:
                    delays.append(backoff_delays[attempt - 1])
                else:
                    return False
        return False