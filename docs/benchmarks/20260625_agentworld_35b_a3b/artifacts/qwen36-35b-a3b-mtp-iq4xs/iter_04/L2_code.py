from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        data = self._jobs[job_id]

        # Ensure data has a place to record delays
        if 'delays' not in data:
            data['delays'] = []

        # Exponential backoff delays: 1s, 2s, 4s
        delays = [1, 2, 4]
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                processor(data)
                return True
            except Exception:
                # Record the delay if there are more attempts to come
                if attempt < max_attempts - 1:
                    data['delays'].append(delays[attempt])

        return False