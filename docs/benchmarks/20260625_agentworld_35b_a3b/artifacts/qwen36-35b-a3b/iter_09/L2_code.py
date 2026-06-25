from typing import Callable

class JobQueue:
    def __init__(self):
        self._jobs = {}

    def add_job(self, job_id: str, data: dict):
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self._jobs.get(job_id)
        if data is None:
            return False

        backoff_delays = [1, 2, 4]
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    data.setdefault('_delays', []).append(backoff_delays[attempt])
                    data['_retry_count'] = attempt + 1
                else:
                    return False
        return False