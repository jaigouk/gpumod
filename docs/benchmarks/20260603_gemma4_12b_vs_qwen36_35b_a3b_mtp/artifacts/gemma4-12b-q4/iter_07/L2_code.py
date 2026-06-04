from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}

        def add_job(self, job_id: str, data: dict):
            self.jobs[job_id] = {"data": data, "retries": 0}

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job_info = self.jobs[job_id]
            max_retries = 3
            backoffs = [1, 2, 4]

            for attempt in range(max_retries + 1):
                try:
                    # Attempt to process the job
                    # Actually, the instructions say "If the processor raises an exception, retry up to 3 times"
                    # So attempt 0 is first try.
                    # Attempt 1 is retry 1.
                    # Attempt 2 is retry 2.
                    # Attempt 3 is retry 3.
                    # Total tries = 4?
                    # Wait: "retry up to 3 times" usually means 1 initial try + 3 retries.
                    # Let's check backoff: 1s, 2s, 4s. That's 3 intervals.
                    # So:
                    # Try 1 -> Fail -> Backoff 1s
                    # Try 2 -> Fail -> Backoff 2s
                    # Try 3 -> Fail -> Backoff 4s
                    # Try 4 -> Fail -> Exhausted.
                    # This matches "retry up to 3 times".

                    # Wait, looking at the requirement:
                    # "retry up to 3 times"
                    # "backoff delays... 1s, 2s, 4s"
                    # This implies:
                    # Initial try
                    # Retry 1 (after 1s)
                    # Retry 2 (after 2s)
                    # Retry 3 (after 4s)
                    # Total attempts = 4.
                    
                    processor(job_info["data"]))
                    # Success
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        job_info["retries"] += 1
                        # Simulation: print or just note the backoff
                        # Requirement says "backoff delays can be stored/tracked rather than actually sleeping"
                        # I'll just track the count and the logic.
                        pass
                    else:
                        return False