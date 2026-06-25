from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job_data = self._jobs[job_id]
        for attempt in range(4):
            try:
                processor(job_data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    job_data.setdefault('delays', []).append(delay)
                    job_data['retry_count'] = attempt + 1
                else:
                    return False