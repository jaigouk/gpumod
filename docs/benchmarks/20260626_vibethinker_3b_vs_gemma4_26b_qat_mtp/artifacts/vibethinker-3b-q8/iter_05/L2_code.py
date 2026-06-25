from typing import Callable
from collections import defaultdict

class JobQueue:
    def __init__(self):
        self._jobs = {}
        self._retry_counts = defaultdict(int)

    def add_job(self, job_id: str, data: dict) -> None:
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self._jobs.get(job_id)
        if not data:
            return False

        for attempt in range(1, 5):
            if attempt > 1:
                delay = 2 ** (attempt - 2)  # 1, 2, 4 seconds
                data.setdefault('delay_log', []).append(delay)
            try:
                processor(data)
                return True
            except Exception:
                self._retry_counts[job_id] += 1
        return False