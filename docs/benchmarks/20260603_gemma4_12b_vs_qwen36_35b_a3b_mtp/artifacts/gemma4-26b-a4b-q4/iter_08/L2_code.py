from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self.jobs.get(job_id)
        if data is None:
            return False

        if 'delays' not in data:
            data['delays'] = []
        if 'retry_count' not in data:
            data['retry_count'] = 0

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    data['delays'].append(delay)
                    data['retry_count'] += 1
                else:
                    return False
        return False