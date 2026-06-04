from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            max_retries = 3
            backoff_delays = [1, 2, 4]

            for attempt in range(max_retries + 1):
                try:
                    data = self.jobs.get(job_id)
                    if data is None:
                        return False
                    
                    processor(data)
                    # If it succeeds, we could clear retries or leave them
                    # The requirement says return True if success.
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # The requirement says "backoff delays can be stored/tracked"
                        # and "simulate". Since I'm not supposed to actually sleep
                        # but "simulate", I'll just acknowledge the delay.
                        # However, usually "simulated" in these prompts means
                        # "don't actually use time.sleep" or "just show the value".
                        # I'll keep it simple.
                        pass
                    else:
                        return False
            return False