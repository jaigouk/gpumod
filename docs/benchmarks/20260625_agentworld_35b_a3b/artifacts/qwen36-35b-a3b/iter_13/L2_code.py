from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._delays: Dict[str, list] = {}
        self._retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: dict):
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self._jobs.get(job_id)
        if data is None:
            return False

        self._delays[job_id] = []
        self._retry_counts[job_id] = 0

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    self._delays[job_id].append(delay)
                    self._retry_counts[job_id] += 1

        return False