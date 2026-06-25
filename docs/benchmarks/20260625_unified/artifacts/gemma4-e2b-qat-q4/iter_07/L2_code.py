import time
from typing import Callable, Dict, Any, List

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}
        self.delays: Dict[str, List[int]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        """Adds a job to the queue."""
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.delays[job_id] = []

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Executes a job processor with a maximum of 4 attempts and exponential backoff.

        Returns:
            True if successful on the first attempt, False otherwise.
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job ID {job_id} not found in queue.")

        data = self.jobs[job_id]
        current_retry = self.retry_counts.get(job_id, 0)

        if current_retry >= 4:
            return False

        try:
            result = processor(data)
            return result
        except Exception:
            # Record attempt and update count
            new_count = current_retry + 1
            self.retry_counts[job_id] = new_count

            if new_count < 4:
                # Calculate exponential backoff: 1s, 2s, 4s
                # Attempt 0 fails -> retry 1 (1s)
                # Attempt 1 fails -> retry 2 (2s)
                # Attempt 2 fails -> retry 3 (4s)
                delay = 2 ** current_retry

                # Requirement 4: Record the delay as data
                if job_id not in self.delays:
                    self.delays[job_id] = []
                self.delays[job_id].append(delay)

                # Requirement 2 & 3: Retry logic continues
                # We don't sleep, we just iterate the process (simulated retry attempt)
                return self.process_job(job_id, processor)
            else:
                # Failed 4 attempts
                return False

# Example Usage (for verification, not part of the required output structure)
if __name__ == "__main__":
    class FailingProcessor:
        def __init__(self, fails_count):
            self.fails_count = 0
        def __call__(self, data):
            self.fails_count += 1
            print(f"Attempt {self.fails_count} for job {data.get('id')}")
            if self.fails_count <= 3:
                raise Exception("Temporary Error")
            return "Success"

    queue = JobQueue()
    test_job = "job_123"
    queue.add_job(test_job, {"id": test_job, "payload": "test_data"})

    # Setup processor to fail 4 times (failing on the first 3 attempts)
    processor_to_test = FailingProcessor(fails_count=4)

    success = queue.process_job(test_job, processor_to_test)
    print(f"\nJob {test_job} process_job result: {success}")

    # Verify delays recorded
    print(f"Delays recorded for {test_job}: {queue.delays.get(test_job)}")