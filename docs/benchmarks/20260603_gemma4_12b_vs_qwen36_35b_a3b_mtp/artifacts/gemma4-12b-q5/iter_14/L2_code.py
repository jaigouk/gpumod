from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]]]) -> bool:
        data = self.jobs.get(job_id)
        if data is None:
            return False

        if 'delays' not in data:
            data['delays']] = []

        # Ensure the retry count starts at 0 for this processing attempt
        self.retry_counts[job_id] = 0

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    data['delays'].append(delay)
                    self.retry_counts[job_id] = attempt + 1
                else:
                    return False
        return False