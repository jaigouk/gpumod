from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict] = {}
        self._retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict):
        self._jobs[job_id] = data.copy()
        self._retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable[[Dict], Any]) -> bool:
        data = self._jobs.get(job_id)
        if not data:
            raise KeyError(f'Job {job_id} not found')
        attempts = 0
        while attempts < 4:
            attempts += 1
            try:
                processor(data)
                return True
            except Exception:
                self._retry_counts[job_id] = attempts
                if attempts == 4:
                    return False
                delay = 2 ** (attempts - 1)
                data[f'delay_{delay}'] = delay