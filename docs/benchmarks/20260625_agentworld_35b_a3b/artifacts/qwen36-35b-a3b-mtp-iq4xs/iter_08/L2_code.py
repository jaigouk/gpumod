from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        max_attempts = 4
        # Exponential backoff delays: 1s, 2s, 4s
        delays = [1, 2, 4]

        for attempt in range(max_attempts):
            try:
                processor(data)
                return True
            except Exception:
                # If we are not on the last attempt, record the delay
                # Attempt 0 fails -> record delay 1 (index 0)
                # Attempt 1 fails -> record delay 2 (index 1)
                # Attempt 2 fails -> record delay 4 (index 2)
                # Attempt 3 fails -> loop ends, no delay recorded
                if attempt < len(delays):
                    delay = delays[attempt]
                    # Record the delay as data
                    if 'retry_delays' not in data:
                        data['retry_delays'] = []
                    data['retry_delays'].append(delay)
                    # Do not actually sleep

        return False