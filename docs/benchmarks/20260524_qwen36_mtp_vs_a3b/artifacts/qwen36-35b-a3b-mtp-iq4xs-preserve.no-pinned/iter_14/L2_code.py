import time # Maybe not needed if we don't sleep, but good for type hinting if we were.
    from typing import Callable, Any, Dict, Optional

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}
            # If we need to store delays, maybe a dict?
            # But the prompt says "Use exponential backoff... 1s, 2s, 4s".
            # This usually implies a formula: 2^attempt.
            # Let's stick to the formula but ensure we track the count.

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False # Or raise error? Let's assume valid input based on example.

            max_retries = 3
            current_retries = self.retry_counts.get(job_id, 0)

            # We need to run the processor.
            # If it fails, we increment retry count, calculate backoff, and retry?
            # Or do we just loop?

            # Wait, the prompt says "Extend the job queue".
            # It implies the queue holds jobs. `process_job` takes a job_id.
            # Usually, a queue processes the *next* item.
            # But the signature `process_job(job_id, processor)` suggests we are targeting a specific job.

            # Let's implement the loop.
            attempts = 0
            while attempts <= max_retries:
                try:
                    job_data = self.jobs[job_id]
                    result = processor(job_data)
                    # Success
                    return True
                except Exception as e:
                    attempts += 1
                    if attempts > max_retries:
                        return False
                    
                    # Backoff logic
                    # 1st retry: wait 1s (2^0)
                    # 2nd retry: wait 2s (2^1)
                    # 3rd retry: wait 4s (2^2)
                    # Wait, the prompt says "retry up to 3 times".
                    # Attempt 1 (Fail) -> Wait 1s -> Attempt 2 (Fail) -> Wait 2s -> Attempt 3 (Fail) -> Wait 4s -> Attempt 4 (Fail) -> Return False.
                    
                    delay = 2 ** (attempts - 1)
                    
                    # "delays can be stored/tracked rather than actually sleeping"
                    # This might mean we just record the delay in the object state?
                    # Or just ignore the sleep?
                    # "can be stored" -> maybe store in a list?
                    # Let's store the pending delay in the retry_counts or a separate dict?
                    # Actually, usually "tracked" means `self.retry_counts[job_id] += 1`.
                    # Let's just update the count.
                    
                    self.retry_counts[job_id] = attempts
                    
                    # To satisfy "delays can be stored", let's store the calculated delay.
                    # But where? Maybe just a local var is enough? 
                    # The prompt is slightly ambiguous. "The backoff delays can be stored/tracked rather than actually sleeping."
                    # This likely means: Don't block the thread. Just increment a counter or set a timestamp for when it *could* run.
                    # However, the method signature returns a bool immediately.
                    # If I don't sleep, the loop runs instantly.
                    # If I don't sleep, the "1s, 2s, 4s" requirement is purely theoretical unless I store it.
                    
                    # Let's store the delays in a list for tracking purposes.
                    if not hasattr(self, 'backoff_delays'):
                        self.backoff_delays = []
                    self.backoff_delays.append(delay)

            return False