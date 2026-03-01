from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Any) -> None:
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0,
            "backoff_delays": []
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        max_retries = 3
        backoff_schedule = [1, 2, 4]

        for attempt in range(max_retries + 1):
            try:
                processor(job["data"])
                return True
            except Exception:
                job["retry_count"] += 1
                if job["retry_count"] <= max_retries:
                    delay_index = job["retry_count"] - 1
                    if delay_index < len(backoff_schedule):
                        job["backoff_delays"].append(backoff_schedule[delay_index])
                else:
                    break
        return False