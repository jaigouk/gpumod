from typing import Callable

class JobQueue:
    def __init__(self):
        self._jobs = {}

    def add_job(self, job_id: str, data: dict):
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self._jobs[job_id]
        delays = [1, 2, 4]
        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    data.setdefault('delays', []).append(delays[attempt])
                    data['retry_count'] = attempt + 1
                else:
                    data['retry_count'] = 4
                    return False
        return False