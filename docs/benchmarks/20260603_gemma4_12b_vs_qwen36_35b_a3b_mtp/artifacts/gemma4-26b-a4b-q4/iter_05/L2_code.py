from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}
            self.retry_history: Dict[str, list] = {} # To track backoffs if needed, but prompt says "Track retry count for each job"

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = {
                "data": data,
                "retries": 0
            }

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            max_retries = 3

            while job["retries"] <= max_retries:
                try:
                    processor(job["data"])
                    return True
                except Exception:
                    if job["retries"] == max_retries:
                        return False
                    
                    # Calculate backoff: 1s, 2s, 4s
                    # Attempt 1 (0 retries) fails -> retry 1 (wait 1s)
                    # Attempt 2 (1 retry) fails -> retry 2 (wait 2s)
                    # Attempt 3 (2 retries) fails -> retry 3 (wait 4s)
                    # Attempt 4 (3 retries) fails -> return False
                    
                    # Let's re-read: "retry up to 3 times".
                    # Initial try (0)
                    # 1st retry (1)
                    # 2nd retry (2)
                    # 3rd retry (3)
                    # Total tries allowed: 4? Or 3 total? Usually "retry up to 3 times" means 1 original + 3 retries = 4 tries.
                    # But the delay sequence is 1s, 2s, 4s. That's 3 delays.
                    # Delay 1 after try 1.
                    # Delay 2 after try 2.
                    # Delay 3 after try 3.
                    # If try 4 fails, we stop.
                    
                    # Implementation:
                    # retry_count starts at 0.
                    # Try processor.
                    # If Exception:
                    #    if retries < 3:
                    #       retries += 1
                    #       wait = 2**(retries-1)
                    #       ...
                    #    else:
                    #       return False

                    # Let's refine the loop.
                    pass