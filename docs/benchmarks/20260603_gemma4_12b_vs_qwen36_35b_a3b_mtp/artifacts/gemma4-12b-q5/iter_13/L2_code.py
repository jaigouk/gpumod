from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self.retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: dict) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable[[dict]]) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        retry_count = self.retry_counts[job_id]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                retry_count += 1
                self.retry_counts[job_id] = retry_count
                if retry_count < 4:
                    delay = 2 ** (retry_count - 1)
                    data['last_delay'] = delay
                else:
                    break

        return False