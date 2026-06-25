from typing import Callable, Dict, Any, List

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        max_attempts = 4
        backoff_delays = [1, 2, 4]
        recorded_delays = []
        retry_count = 0

        for attempt in range(max_attempts):
            try:
                processor(data)
                return True
            except Exception:
                retry_count += 1
                # Record delay if there is a subsequent retry
                if retry_count <= len(backoff_delays):
                    recorded_delays.append(backoff_delays[retry_count - 1])

        # Record delays and retry count in job data
        data['delays'] = recorded_delays
        data['retry_count'] = retry_count
        return False