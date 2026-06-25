from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        # Stores the number of times the job has been attempted (including the first attempt)
        self.job_attempts: Dict[str, int] = {}
        # Stores the recorded delays for each job: {job_id: [delay1, delay2, ...]}
        self.recorded_delays: Dict[str, list[float]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        """Adds a job to the queue."""
        self.jobs[job_id] = data
        self.job_attempts[job_id] = 0
        self.recorded_delays[job_id] = []

    def process_job(self, job_id: str, processor: Callable[..., Any]) -> bool:
        """
        Attempts to process a job with retries and exponential backoff.

        Args:
            job_id: The ID of the job.
            processor: The callable function to execute on the job data.

        Returns:
            True if successful on any attempt, False if all attempts fail.
        """
        if job_id not in self.jobs:
            return False

        job_data = self.jobs[job_id]
        max_attempts = 4

        for attempt in range(max_attempts):
            # Increment attempt counter (0 is initial attempt, 1 is retry 1, etc.)
            self.job_attempts[job_id] = attempt

            try:
                # 1. Call processor
                processor(job_data)
                # 5. Return True on success
                return True

            except Exception as e:
                # If successful, we return True above.
                # If failed, record the attempt and check if we should retry.

                # 3. Track retry count (done by the loop counter 'attempt')

                if attempt < max_attempts - 1:
                    # 2. Retry is possible. Calculate backoff delay.
                    # 3. Exponential backoff: 1s (attempt 0 -> 1), 2s (attempt 1 -> 2), 4s (attempt 2 -> 3)
                    delay = 2 ** attempt

                    # 4. Record the delay (do NOT actually sleep)
                    if job_id not in self.recorded_delays:
                        self.recorded_delays[job_id] = []
                    self.recorded_delays[job_id].append(delay)

                    # 3. Continue to the next attempt
                    continue

        # If the loop finishes without returning True
        # 5. Return False if all 4 attempts raise
        return False

# Example Usage (Optional, for testing logic):
if __name__ == '__main__':
    import time

    queue = JobQueue()

    # --- Example 1: Job that succeeds on the second try ---
    def retry_target_success(data):
        print(f"Processing job {data['id']} (Attempt {queue.job_attempts.get('test_job', 0)})")
        if queue.job_attempts.get('test_job', 0) < 2:
            raise ValueError("Transient failure")
        print("Success!")

    queue.add_job("test_job", {"id": "test_job"})
    success1 = queue.process_job("test_job", retry_target_success)
    print(f"\nTest 1 Result (Expected True): {success1}")
    print(f"Recorded Delays for test_job: {queue.recorded_delays['test_job']}")


    # --- Example 2: Job that fails all attempts ---
    def retry_target_always_fail(data):
        print(f"Processing job {data['id']} (Attempt {queue.job_attempts.get('fail_job', 0)})")
        raise RuntimeError("Permanent failure")

    queue.add_job("fail_job", {"id": "fail_job"})
    success2 = queue.process_job("fail_job", retry_target_always_fail)
    print(f"\nTest 2 Result (Expected False): {success2}")
    print(f"Recorded Delays for fail_job: {queue.recorded_delays['fail_job']}")

    # Verify attempt count tracking
    print("\n--- Tracking ---")
    print(f"Final Attempts for test_job: {queue.job_attempts['test_job']}")
    print(f"Final Attempts for fail_job: {queue.job_attempts['fail_job']}")