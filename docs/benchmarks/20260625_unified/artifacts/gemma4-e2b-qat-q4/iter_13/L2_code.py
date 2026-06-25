from typing import Callable, Dict, Any

class JobQueue:
    """
    A simple job queue implementation supporting retry logic.
    """
    def __init__(self):
        # Stores {job_id: data}
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        """Adds a job to the queue."""
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Processes a job with retries using exponential backoff.

        Retries are limited to 4 attempts. Delays are calculated but not executed.

        Args:
            job_id: The identifier of the job to process.
            processor: The callable function to execute on the job data.

        Returns:
            True if the job succeeded, False otherwise.
        """
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]
        max_attempts = 4

        for attempt in range(1, max_attempts + 1):
            try:
                print(f"Attempt {attempt}/{max_attempts} for job {job_id}...")

                # 1. Call processor(data)
                processor(data)

                # Return True on success
                return True

            except Exception as e:
                if attempt < max_attempts:
                    # 3. Track retry count and 4. Calculate exponential backoff delay
                    # Delays: attempt 1 -> 1s (for attempt 2); attempt 2 -> 2s (for attempt 3); attempt 3 -> 4s (for attempt 4)
                    # The delay calculation is based on the number of failed attempts so far (attempt - 1).

                    # Calculation: 2^(attempt - 2) * 1 (for attempt 1 failure)
                    # Attempt 1 failure (going to attempt 2) -> delay = 1
                    # Attempt 2 failure (going to attempt 3) -> delay = 2
                    # Attempt 3 failure (going to attempt 4) -> delay = 4

                    backoff_delay = 2**(attempt - 2)

                    # 4. Do NOT actually sleep — record the delays as data.
                    print(f"Job {job_id} failed. Retrying in {backoff_delay}s (Attempt {attempt}) due to: {e}")

                    # In a real implementation, we would sleep here: time.sleep(backoff_delay)
                else:
                    # All 4 attempts failed
                    print(f"Job {job_id} failed after {max_attempts} attempts. Final error: {e}")
                    return False

        return False

# Example Usage (for verification, not part of the required output structure):
# q = JobQueue()
# q.add_job("job_A", {"payload": "test"})
#
# def failing_processor(data):
#     if not hasattr(failing_processor, 'count'):
#         failing_processor.count = 0
#     failing_processor.count += 1
#     if failing_processor.count <= 2:
#         raise ValueError("Transient error")
#     raise RuntimeError("Permanent error")
#
# q.process_job("job_A", failing_processor)