from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = {
            **data,
            'retry_count': 0,
            'backoff_log': []
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job_data = self.jobs[job_id]
        max_retries = 3
        backoff_schedule = [1, 2, 4]
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = backoff_schedule[attempt - 1]
                job_data['backoff_log'].append(delay)
                job_data['retry_count'] += 1
            
            try:
                processor(job_data)
                return True
            except Exception:
                if attempt == max_retries:
                    break
        
        return False