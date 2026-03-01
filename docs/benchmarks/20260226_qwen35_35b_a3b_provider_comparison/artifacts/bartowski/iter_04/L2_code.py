from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Any):
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0,
            "backoff_delays": []
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        data = job["data"]
        max_retries = 3
        
        # Attempt initial run + up to 3 retries
        for attempt in range(max_retries + 1):
            try:
                processor(data)
                return True
            except Exception:
                job["retry_count"] += 1
                
                if job["retry_count"] > max_retries:
                    return False
                
                # Exponential backoff: 1s, 2s, 4s
                # Retry 1 -> 2^0 = 1s
                # Retry 2 -> 2^1 = 2s
                # Retry 3 -> 2^2 = 4s
                delay = 2 ** (job["retry_count"] - 1)
                job["backoff_delays"].append(delay)

        return False