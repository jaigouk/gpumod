from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

        def add_job(self, job_id: str, data: Dict[str, Any]):
            self.jobs[job_id] = {"data": data, "retries": 0}

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            max_retries = 3

            while job["retries"] <= max_retries:
                try:
                    processor(job["data"])
                    return True
                except Exception as e:
                    job["retries"] += 1
                    if job["retries"] > max_retries:
                        break
                    
                    # Exponential backoff calculation: 1, 2, 4
                    delay = 2 ** (job["retries"] - 1)
                    print(f"Retry {job['retries']} for {job_id} after {delay}s delay due to: {e}")
                    # No actual sleep as per requirements
            
            return False