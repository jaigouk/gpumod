from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0,
            "backoff_delay": 0
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        job_data = job["data"]
        max_retries = 3
        base_delay = 1
        
        # Total attempts = 1 initial + max_retries
        for attempt in range(max_retries + 1):
            try:
                processor(job_data)
                # Success: reset retry count
                job["retry_count"] = 0
                job["backoff_delay"] = 0
                return True
            except Exception:
                job["retry_count"] += 1
                current_retries = job["retry_count"]
                
                if current_retries <= max_retries:
                    # Calculate exponential backoff: 1s, 2s, 4s
                    delay = base_delay * (2 ** (current_retries - 1))
                    job["backoff_delay"] = delay
                    # Simulate sleep by storing delay instead of time.sleep()
                    # time.sleep(delay) 
                else:
                    # Max retries exhausted
                    return False
                    
        return False