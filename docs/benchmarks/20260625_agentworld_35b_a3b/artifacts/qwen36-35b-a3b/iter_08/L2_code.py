from typing import Callable

class JobQueue:
    def __init__(self):
        self._jobs = {}
        self._delays = {}

    def add_job(self, job_id: str, data: dict):
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self._jobs.get(job_id)
        if data is None:
            return False

        delays = []
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    delays.append(delay)

        self._delays[job_id] = delays
        return False