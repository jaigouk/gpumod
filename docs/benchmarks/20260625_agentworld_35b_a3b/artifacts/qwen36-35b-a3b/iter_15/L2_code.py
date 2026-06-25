from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self._jobs[job_id]
        max_attempts = 4
        backoff_delays = [1, 2, 4]

        for attempt in range(1, max_attempts + 1):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < max_attempts:
                    delay = backoff_delays[attempt - 1]
                    data.setdefault('delays', []).append(delay)
                    data['retry_count'] = attempt
        return False