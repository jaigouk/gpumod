import time # Optional, but maybe not needed if we just track
    from typing import Callable, Any, Dict, Optional

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}
            self.retry_counts: Dict[str, int] = {}
            self.backoff_delays: Dict[str, list] = {}
            # Or calculate backoff dynamically based on retry count
            self.max_retries = 3
            self.backoff_sequence = [1, 2, 4] # seconds

        def add_job(self, job_id: str, data: Dict[str, Any]):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0
            # Initialize backoff tracking if needed

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False # Or handle error

            data = self.jobs[job_id]
            retry_count = self.retry_counts.get(job_id, 0)
            max_retries = self.max_retries
            backoff_delays = self.backoff_sequence
            # Calculate current backoff delay based on retry count
            # Retry 0 (Initial) -> No wait
            # Retry 1 -> Wait backoff[0]
            # Retry 2 -> Wait backoff[1]
            # Retry 3 -> Wait backoff[2]

            # Loop logic
            # We need to distinguish between initial attempt and retries
            # Let's use a loop range(max_retries + 1)
            
            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    return True
                except Exception:
                    self.retry_counts[job_id] = attempt + 1
                    if attempt < max_retries:
                        # Calculate backoff
                        delay_idx = attempt # 0 -> 1s, 1 -> 2s, 2 -> 4s
                        current_delay = backoff_delays[delay_idx]
                        # Store it? The prompt says "backoff delays can be stored/tracked"
                        # Let's just track the fact that we waited or plan to wait.
                        # For simplicity, I'll just record the delay in the state if needed,
                        # but the requirement mainly says "tracked".
                        # I will store the delay in a separate dict to satisfy "tracked".
                        if job_id not in self.backoff_delays:
                            self.backoff_delays[job_id] = []
                        self.backoff_delays[job_id].append(current_delay)
                    else:
                        return False
            
            return False # Should be covered by except block above