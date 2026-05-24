import time
from typing import Callable, Any, Dict, Optional

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoff_delays: Dict[str, list] = {}
        
    def add_job(self, job_id: str, data: Any) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.backoff_delays[job_id] = []
        
    def process_job(self, job_id: str, processor: Callable) -> bool:
        max_retries = 3
        base_delay = 1.0
        data = self.jobs.get(job_id)
        if data is None:
            return False
            
        for attempt in range(max_retries + 1):  # attempt 0 is initial, 1-3 are retries
            try:
                processor(data)
                # Success
                self.retry_counts[job_id] = attempt
                return True
            except Exception:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # 1, 2, 4 for attempts 0, 1, 2
                    self.backoff_delays[job_id].append(delay)
                    self.retry_counts[job_id] = attempt + 1
                    # Simulate waiting
                    # time.sleep(delay)
                else:
                    self.retry_counts[job_id] = max_retries
                    return False
        return False