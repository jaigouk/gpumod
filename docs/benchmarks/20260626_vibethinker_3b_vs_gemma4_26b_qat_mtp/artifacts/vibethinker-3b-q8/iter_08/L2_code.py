from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Any] = {}
        self._retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict) -> None:
        self._jobs[job_id] = data
        if job_id not in self._retry_counts:
            self._retry_counts[job_id] = 1

    def process_job(self, job_id: str, processor: Callable[[Dict], Any]) -> bool:
        data = self._jobs.get(job_id)
        if data is None:
            raise KeyError(f"Job {job_id} not found")
        attempt = 1
        max_attempts = 4
        while attempt <= max_attempts:
            try:
                if attempt > 1:
                    delay = 2 ** (attempt - 2)
                    data[f'delay_{attempt-1}'] = delay
                processor(data)
                return True
            except Exception:
                attempt += 1
                if attempt > max_attempts:
                    return False
                self._retry_counts[job_id] = attempt
        return False