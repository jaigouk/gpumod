from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict) -> None:
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        data = self._jobs[job_id]
        delays = [1, 2, 4]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    data.setdefault('retry_delays', []).append(delays[attempt])
                data['retry_count'] = attempt + 1

        return False