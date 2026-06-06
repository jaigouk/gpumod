from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        max_attempts = 4
        delays = [1, 2, 4]

        for attempt in range(max_attempts):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < max_attempts - 1:
                    if 'delays' not in data:
                        data['delays'] = []
                    data['delays'].append(delays[attempt])

        return False