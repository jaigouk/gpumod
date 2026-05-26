import time
    from typing import Callable, Any, Dict

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}
            self.backoff_delays = [1, 2, 4] # 1s, 2s, 4s
            self.max_retries = 3

        def add_job(self, job_id: str, data: Any) -> None:
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            current_count = self.retry_counts.get(job_id, 0)
            # If we just process it, do we reset?
            # Let's assume process_job handles the full lifecycle or continues from state.
            # To keep it simple based on example: process_job runs the logic.
            # However, "Track retry count for each job" implies we store it.
            # Let's reset to 0 at start of process_job for a fresh run?
            # Or increment on failure.
            # Let's increment on failure.
            
            # Actually, if I call process_job, it should attempt.
            # Let's track retries within this call.
            
            # Reset count for this specific process_job execution logic?
            # Or persist? "Track retry count for each job" usually means metadata.
            # I will persist it in self.retry_counts.
            # Reset to 0 at start of processing?
            # Let's assume process_job is the entry point for execution.
            # So start with 0 retries for this execution.
            
            retries = 0
            attempts = 0
            max_attempts = self.max_retries + 1 # 1 initial + 3 retries
            
            while attempts < max_attempts:
                try:
                    processor(self.jobs[job_id])
                    self.retry_counts[job_id] = retries
                    return True
                except Exception:
                    retries += 1
                    attempts += 1
                    
                    if retries <= len(self.backoff_delays):
                        delay = self.backoff_delays[retries - 1]
                        time.sleep(delay)
                    else:
                        return False
            
            self.retry_counts[job_id] = retries
            return False