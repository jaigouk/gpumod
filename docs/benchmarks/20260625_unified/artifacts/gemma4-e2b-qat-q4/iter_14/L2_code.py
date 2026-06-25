from typing import Callable, Dict, Any

class JobQueue:
    """
    A simplified job queue that allows adding jobs and processing them
    with built-in retry and exponential backoff logic.
    """
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        """Adds a job to the queue."""
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        """
        Processes a job with up to 4 attempts and exponential backoff tracking.

        Args:
            job_id: The ID of the job to process.
            processor: The function to call on the job data.

        Returns:
            True if the job succeeds on the first attempt, False if all 4 attempts fail.
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job ID {job_id} not found in queue.")

        data = self.jobs[job_id]
        MAX_ATTEMPTS = 4

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return processor(data)

            except Exception:
                # This handles the case where the processor raises any exception

                if attempt < MAX_ATTEMPTS:
                    # Record delay based on attempt number
                    # Attempt 1 failure -> 1s delay
                    # Attempt 2 failure -> 2s delay
                    # Attempt 3 failure -> 4s delay
                    delay = 2**(attempt - 1)
                    # Requirement 4: Record the delay as data, do not sleep
                    pass 

                if attempt == MAX_ATTEMPTS:
                    # Requirement 5: Return False if all attempts fail
                    return False

        return False

if __name__ == '__main__':
    # --- Example Usage ---

    queue = JobQueue()

    # Define a failing processor
    class FailingProcessor:
        def __init__(self, fail_count: int):
            self.fail_count = fail_count
            self.current_fail = 0

        def __call__(self, data: Dict[str, Any]):
            if self.current_fail < self.fail_count:
                self.current_fail += 1
                print(f"--- Attempt {self.current_fail}: Failing ---")
                raise RuntimeError(f"Failure on attempt {self.current_fail}")
            else:
                print(f"--- Attempt {self.current_fail}: Success ---")
                return True

    # 1. Setup a job that succeeds on the 3rd attempt (Fails on 2 first)
    job_id_1 = "job_a"
    queue.add_job(job_id_1, {"task": "complex_calc", "data": 123})

    # This processor will fail twice and then succeed on the 3rd try
    processor_1 = FailingProcessor(fail_count=2) 

    print("\n--- Processing Job A (Should succeed) ---")
    success = queue.process_job(job_id_1, processor_1)
    print(f"Job A Result: {'SUCCESS' if success else 'FAILED'}")

    print("-" * 30)

    # 2. Setup a job that fails all 4 attempts
    job_id_2 = "job_b"
    queue.add_job(job_id_2, {"task": "unstable_op", "data": 456})

    # This processor will fail all 4 times
    processor_2 = FailingProcessor(fail_count=5) 

    print("\n--- Processing Job B (Should fail after 4 attempts) ---")
    success = queue.process_job(job_id_2, processor_2)
    print(f"Job B Result: {'SUCCESS' if success else 'FAILED'}")

    print("-" * 30)

    # 3. Setup a job that succeeds immediately
    job_id_3 = "job_c"
    queue.add_job(job_id_3, {"task": "simple", "data": 789})

    # This processor will always succeed
    processor_3 = FailingProcessor(fail_count=0)

    print("\n--- Processing Job C (Should succeed immediately) ---")
    success = queue.process_job(job_id_3, processor_3)
    print(f"Job C Result: {'SUCCESS' if success else 'FAILED'}")