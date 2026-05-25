import time
from typing import Callable, Any, Dict, Optional

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0,
            "next_backoff": 1.0
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False
            
        job = self.jobs[job_id]
        max_retries = 3
        
        for attempt in range(max_retries + 1):  # 0 to 3 -> 4 attempts total? Wait, "retry up to 3 times" usually means 1 initial + 3 retries = 4 attempts, or just 3 attempts total. Let's assume 3 retries means 4 attempts total. Or maybe 3 attempts total. I'll stick to 3 retries = 4 attempts. Actually, standard is often retries=3 means 3 extra attempts. I'll do range(max_retries + 1).
            try:
                processor(job["data"])
                return True
            except Exception as e:
                if attempt < max_retries:
                    job["retry_count"] += 1
                    job["next_backoff"] = job["next_backoff"] * 2
                    # Simulate backoff by storing it, not sleeping
                    # Or just continue
                    continue
                else:
                    return False
        return False