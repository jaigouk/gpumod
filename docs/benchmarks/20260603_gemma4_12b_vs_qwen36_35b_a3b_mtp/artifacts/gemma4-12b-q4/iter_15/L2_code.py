from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        data['retries'] = 0
        data['delays'] = []
        self.jobs[job_id] = data

    def process_job(self, job_id: str, Processor: Callable[[Dict[str, Any]]]) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        backoff_delays = [1, 2, 4]
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                Processor(data)
                return True
            except Exception:
                if attempt < max_attempts - 1:
                    delay = backoff_delays[attempt]
                    data['retries'] += 1
                    data['delays'].append(delay)
                else:
                    break

        return False