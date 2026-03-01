from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, args: Dict[str, Any]) -> None:
        self.jobs[job_id] = {
            "args": args,
            "retry_count": 0,
            "max_retries": 3
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        args = job["args"]
        max_retries = job["max_retries"]
        
        current_attempt = 0
        while current_attempt <= max_retries:
            try:
                processor(args)
                # Reset retry count on success
                job["retry_count"] = 0
                return True
            except Exception:
                current_attempt += 1
                job["retry_count"] = current_attempt
                
                if current_attempt <= max_retries:
                    # Exponential backoff: 1s, 2s, 4s
                    backoff = 2 ** (current_attempt - 1)
                    job["last_backoff"] = backoff
                    # Backoff delays are tracked rather than actually sleeping
                    # time.sleep(backoff) 
                else:
                    return False
        
        return False