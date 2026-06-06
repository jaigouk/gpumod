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
        backoffs = [1, 2, 4]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    # Record retry information in the data dictionary
                    data['retry_count'] = attempt + 1
                    data['delay'] = backoffs[attempt]
                else:
                    # All 4 attempts (initial + 3 retries) failed
                    return False
        return False