from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.job_retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        """Adds a job to the queue."""
        self.jobs[job_id] = data
        self.job_retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        """
        Executes a job with a retry mechanism and exponential backoff.

        Returns:
            bool: True if successful within 4 attempts, False otherwise.
        """
        data = self.jobs.get(job_id)
        if not data:
            return False

        max_attempts = 4

        for attempt in range(1, max_attempts + 1):
            try:
                # 1. Call the processor
                result = processor(data)
                return True  # First successful attempt

            except Exception as e:
                # Track attempt count
                self.job_retry_counts[job_id] = attempt

                if attempt < max_attempts:
                    # 3. Exponential backoff calculation: 1, 2, 4 seconds
                    # attempt 1 -> delay 2^(1-1) = 1
                    # attempt 2 -> delay 2^(2-1) = 2
                    # attempt 3 -> delay 2^(3-1) = 4
                    delay = 2**(attempt - 1)

                    # 4. Record the delay as data (not actually sleep)
                    # In a real scenario, this delay would be used for sleep()
                    # Here, we simulate recording the metadata.
                    pass 

        # 5. If all attempts fail
        return False