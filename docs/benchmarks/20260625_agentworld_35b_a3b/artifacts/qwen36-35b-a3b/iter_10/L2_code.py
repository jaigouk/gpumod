from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
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
                    delay = backoff_delays[attempt]
                    data.setdefault('delays', []).append(delay)
                    data['retry_count'] = attempt
        return False