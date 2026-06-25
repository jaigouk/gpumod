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
        backoff_schedule = [1, 2, 4]
        recorded_delays = []

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = backoff_schedule[attempt]
                    recorded_delays.append(delay)
                    data['retry_count'] = attempt + 1
                    data['delays'] = recorded_delays
                else:
                    return False
        return False