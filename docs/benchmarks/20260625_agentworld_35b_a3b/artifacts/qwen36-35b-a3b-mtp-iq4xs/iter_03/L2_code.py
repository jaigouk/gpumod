from typing import Callable, Any, Dict, List

class JobQueue:
    def __init__(self):
        self._jobs = {}

    def add_job(self, job_id: str, data: dict):
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        data = self._jobs[job_id]
        # Exponential backoff delays for retries: 1s, 2s, 4s
        backoff_delays = [1, 2, 4]

        for attempt in range(4):
            # If this is a retry (attempt > 0), record the delay
            if attempt > 0:
                delay = backoff_delays[attempt - 1]
                # Record the delay as data
                if 'delays' not in data:
                    data['delays'] = []
                data['delays'].append(delay)
                # Requirement: Do NOT actually sleep

            try:
                processor(data)
                return True
            except Exception:
                # Exception occurred, proceed to next attempt
                pass

        return False