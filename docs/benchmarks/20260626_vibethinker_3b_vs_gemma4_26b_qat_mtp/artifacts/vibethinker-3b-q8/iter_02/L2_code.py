from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs = {}                # job_id -> data dict
        self._retry_counter = {}      # job_id -> attempt count

    def add_job(self, job_id: str, data: Dict) -> None:
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable[[Dict], Any]) -> bool:
        data = self._jobs.get(job_id)
        if data is None:
            raise KeyError(f"Job {job_id} not found")
        attempts = 1
        self._retry_counter[job_id] = attempts
        while attempts <= 4:
            try:
                processor(data)
                return True
            except Exception:
                if attempts == 4:
                    return False
                # exponential backoff: 1, 2, 4 seconds for the 2nd, 3rd, 4th attempts
                delay = 2 ** (attempts - 1)
                # record the delay in the data (no actual sleep)
                data.setdefault('delays', []).append(delay)
                attempts += 1
                self._retry_counter[job_id] = attempts
        return False