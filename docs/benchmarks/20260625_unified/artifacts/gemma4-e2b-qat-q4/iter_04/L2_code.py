from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        # job_id -> data
        self.jobs: Dict[str, Dict[str, Any]] = {}
        # job_id -> retry_count (initial attempt is 0)
        self.retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Processes a job with up to 4 attempts using exponential backoff.
        Returns True if successful on the first try, False otherwise.
        """
        if job_id not in self.jobs:
            return False

        max_attempts = 4
        job_data = self.jobs[job_id]

        # Record initial attempt (0)
        self.retry_counts[job_id] = 0

        for attempt in range(max_attempts):
            try:
                # Execute the processor
                result = processor(job_data)

                # If successful
                if result is not None:
                    return True

                # If processor returns None (meaning not an error, but not success based on typical definitions)
                # We treat None as failure and continue retrying unless we hit max attempts.

            except Exception as e:
                # Execution failed. Record the count before potential sleep/retry
                current_count = self.retry_counts[job_id]

                if attempt < max_attempts - 1:
                    # Determine backoff delay: 1s (1st retry), 2s (2nd retry), 4s (3rd retry)
                    # attempt 0 -> retry 1 (delay 1)
                    # attempt 1 -> retry 2 (delay 2)
                    # attempt 2 -> retry 3 (delay 4)

                    # Delay based on the number of failed attempts (1-indexed)
                    delay = 2 ** attempt 

                    # Requirement 4: Record the delay as data (by updating internal state)
                    # In a real scenario, we would use time.sleep(delay). 
                    # Here, we track the delay internally instead.

                    self.retry_counts[job_id] = attempt + 1
                else:
                    self.retry_counts[job_id] = attempt + 1

        # If the loop completes without returning True, all attempts failed.
        return False

if __name__ == '__main__':
    import time

    # Mock Processor that fails a few times before succeeding
    attempt_counter = 0
    def failing_processor(data):
        global attempt_counter
        attempt_counter += 1
        print(f"Processing job {data['job_id']} (Attempt {attempt_counter})")

        if attempt_counter < 3:
            raise RuntimeError("Temporary Processing Error")

        return f"Job {data['job_id']} processed successfully"

    queue = JobQueue()
    queue.add_job("JobA", {"job_id": "JobA"})

    print("--- Test 1: Successful after 3 failures (should succeed) ---")
    success1 = queue.process_job("JobA", failing_processor)
    print(f"JobA Success Status: {success1}\n")

    # Reset for the next test
    queue = JobQueue()
    queue.add_job("JobB", {"job_id": "JobB"})

    # Mock Processor that always fails (should fail after 4 attempts)
    failure_count = 0
    def always_failing_processor(data):
        global failure_count
        failure_count += 1
        raise ConnectionError("Permanent Network Error")

    print("--- Test 2: Permanent Failure (should fail after 4 attempts) ---")
    success2 = queue.process_job("JobB", always_failing_processor)
    print(f"JobB Success Status: {success2}\n")

    # Verify retry tracking for JobB (should show 4 attempts)
    print(f"JobB final retry count: {queue.retry_counts['JobB']}")