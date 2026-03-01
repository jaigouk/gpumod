from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoff_delays: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.backoff_delays[job_id] = 0

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        max_retries = 3
        backoff_sequence = [1, 2, 4]
        retry_count = 0

        while retry_count <= max_retries:
            try:
                processor(self.jobs[job_id])
                return True
            except Exception:
                retry_count += 1
                self.retry_counts[job_id] = retry_count

                if retry_count <= max_retries:
                    delay_index = retry_count - 1
                    if delay_index < len(backoff_sequence):
                        delay = backoff_sequence[delay_index]
                        self.backoff_delays[job_id] = delay
                    else:
                        self.backoff_delays[job_id] = 0
                else:
                    self.backoff_delays[job_id] = 0
                    break

        return False