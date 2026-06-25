from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        # Exponential backoff delays: 1s, 2s, 4s
        delays = [1, 2, 4]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                # If this was not the last attempt (indices 0, 1, 2 allow retry)
                if attempt < 3:
                    # Record delay as data
                    delay = delays[attempt]
                    if 'delays' not in data:
                        data['delays'] = []
                    data['delays'].append(delay)

                    # Track retry count
                    data['retry_count'] = data.get('retry_count', 0) + 1

                    # Do not sleep
                else:
                    # All 4 attempts failed
                    return False
        return False