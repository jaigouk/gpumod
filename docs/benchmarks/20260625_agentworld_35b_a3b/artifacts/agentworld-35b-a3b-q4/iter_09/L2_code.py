from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict):
        if job_id not in self.jobs:
            self.jobs[job_id] = {'data': data, 'delays': []}
        else:
            self.jobs[job_id]['data'] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job_record = self.jobs.get(job_id)
        if job_record is None:
            raise KeyError(f"Job {job_id} not found")
        
        data = job_record['data']
        delays = job_record.setdefault('delays', [])
        
        for attempt_num in range(1, 5):
            try:
                processor(data)
                return True
            except Exception:
                if attempt_num == 4:
                    break
                delay = 2**(attempt_num - 1)
                delays.append(delay)
        
        return False