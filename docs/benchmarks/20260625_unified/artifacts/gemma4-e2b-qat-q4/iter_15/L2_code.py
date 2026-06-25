import time
from typing import Callable, Dict, Any

class JobQueue:
    """
    A simple queue management system supporting job processing with exponential backoff retries.
    """
    def __init__(self):
        # Stores job data and retry counts: {job_id: {"data": dict, "retries": int}}
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict):
        """Adds a job to the queue."""
        if job_id in self.jobs:
            raise ValueError(f"Job ID {job_id} already exists.")
        self.jobs[job_id] = {"data": data, "retries": 0}

    def process_job(self, job_id: str, processor: Callable[[dict], Any]) -> bool:
        """
        Attempts to process a job with up to 4 retries using exponential backoff delays.

        Args:
            job_id: The identifier of the job.
            processor: A callable that takes the job data dict.

        Returns:
            True if the job was successfully processed, False otherwise.
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job ID {job_id} not found.")

        job_info = self.jobs[job_id]
        data = job_info["data"]
        max_attempts = 4

        for attempt in range(max_attempts):
            # 1. Call processor
            try:
                processor(data)
                # Success
                return True
            except Exception as e:
                # 2. On any exception from the processor, retry.
                # 3. Track the retry count per job.

                job_info["retries"] += 1
                current_attempt = attempt + 1

                if current_attempt < max_attempts:
                    # 4. Exponential backoff: 1s, 2s, 4s delays (based on retries 1, 2, 3)
                    delay = 2 ** (current_attempt - 1)
                    # 5. Do NOT actually sleep — record the delays as data (for tracking purposes)
                    print(f"Job {job_id} failed on attempt {current_attempt}. Retrying in {delay}s...")
                    # In a real system, we would 'time.sleep(delay)' here.
                else:
                    print(f"Job {job_id} failed after {max_attempts} attempts.")
                    # If this is the final failed attempt, the loop exits and returns False.
                    break

        # Return False if all attempts raise
        return False

if __name__ == "__main__":
    # --- Example Usage ---

    q = JobQueue()

    def failing_processor_1(data):
        print(f"Attempt 1: Processing data {data}. Failure simulated.")
        raise RuntimeError("Initial failure")

    def failing_processor_2(data):
        print(f"Attempt 2: Processing data {data}. Failure simulated.")
        raise ValueError("Processing error")

    def successful_processor(data):
        print(f"Attempt 1: Processing data {data}. Success!")
        return True

    # Test Case 1: Job that fails twice and succeeds on the third attempt (2 retries)
    q.add_job("job_001", {"task": "database_op"})

    def processor_success_on_3rd_try(data):
        job_id = "job_001"
        if job_id == "job_001":
            if q.jobs["job_001"]["retries"] < 2:
                print(f"Attempt {q.jobs['job_001']['retries'] + 1}: Processing {job_id}. Failure.")
                raise ConnectionError("Temporary connection issue")
            else:
                print(f"Attempt {q.jobs['job_001']['retries'] + 1}: Processing {job_id}. Success!")
                return True

    print("--- Running Test Case 1: Success on 3rd attempt ---")
    result1 = q.process_job("job_001", processor_success_on_3rd_try)
    print(f"Result 1: {'Success' if result1 else 'Failed'}")
    print(f"Job 001 final retries tracked: {q.jobs['job_001']['retries']}\n")

    # Test Case 2: Job that fails all 4 attempts
    q.add_job("job_002", {"task": "critical_op"})

    def failing_processor_constant(data):
        raise RuntimeError("Permanent error")

    print("--- Running Test Case 2: Failure on all 4 attempts ---")
    result2 = q.process_job("job_002", failing_processor_constant)
    print(f"Result 2: {'Success' if result2 else 'Failed'}")
    print(f"Job 002 final retries tracked: {q.jobs['job_002']['retries']}")