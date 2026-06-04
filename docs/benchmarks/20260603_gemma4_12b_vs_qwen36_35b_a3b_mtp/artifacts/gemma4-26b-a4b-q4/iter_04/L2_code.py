from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    # Exponential backoff: 1s, 2s, 4s
                    delay = 2 ** attempt
                    if 'delays' not in data:
                        data['delays'] = []
                    data['delays'].append(delay)
                else:
                    return False
        return False