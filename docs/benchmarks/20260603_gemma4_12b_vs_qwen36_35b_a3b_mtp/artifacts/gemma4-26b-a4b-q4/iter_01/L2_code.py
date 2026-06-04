from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = {"data": data, "retries": 0}

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            
            while job["retries"] <= 3:
                try:
                    processor(job["data"])
                    return True
                except Exception:
                    if job["retries"] == 3:
                        return False
                    
                    # Backoff: 1s, 2s, 4s
                    # retry 0 failure -> delay 2^0 = 1
                    # retry 1 failure -> delay 2^1 = 2
                    # retry 2 failure -> delay 2^2 = 4
                    # retry 3 failure -> break
                    
                    # The prompt says "The backoff delays can be stored/tracked rather than actually sleeping."
                    # I'll just increment the retry count.
                    job["retries"] += 1
                    # In a real app: time.sleep(2**(job["retries"]-1))
            
            return False