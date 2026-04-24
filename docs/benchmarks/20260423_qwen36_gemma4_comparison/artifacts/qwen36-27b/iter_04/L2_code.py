from typing import Callable, Any, Dict
from collections import defaultdict

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Any] = {}
        self._retry_counts: Dict[str, int] = defaultdict(int)
        self._backoff_delays: Dict[str, list] = defaultdict(list)

    def add_job(self, job_id: str, data: Any) -> None:
        self._jobs[job_id] = data
        self._retry_counts[job_id] = 0
        self._backoff_delays[job_id] = []

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        max_retries = 3
        self._retry_counts[job_id] = 0
        self._backoff_delays[job_id] = []

        for attempt in range(max_retries + 1):
            try:
                processor(self._jobs[job_id])
                self._retry_counts[job_id] = 0
                self._backoff_delays[job_id] = []
                return True
            except Exception:
                if attempt < max_retries:
                    delay = 2 ** attempt
                    self._backoff_delays[job_id].append(delay)
                    self._retry_counts[job_id] = attempt + 1
                else:
                    return False