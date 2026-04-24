import time
from typing import Callable, Dict, Any, Optional

class JobQueue:
    """
    A job queue implementation featuring retry logic and exponential backoff.
    """
    def __init__(self):
        # Stores job data: {job_id: {"data": Any, "retries": int}}
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self.MAX_RETRIES = 3

    def add_job(self, job_id: str, data: Dict[str, Any]):
        """
        Adds a new job to the queue.
        """
        if job_id not in self._jobs:
            self._jobs[job_id] = {"data": data, "retries": 0}
        else:
            print(f"Warning: Job ID {job_id} already exists.")

    def _calculate_backoff_delay(self, attempt: int) -> int:
        """
        Calculates the exponential backoff delay (1s, 2s, 4s, ...).
        attempt starts from 0 for the first failure (first retry).
        """
        # Delay = 2^attempt seconds
        return 2 ** attempt

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        """
        Processes a job with retry logic and exponential backoff.

        Args:
            job_id: The ID of the job to process.
            processor: The function that executes the job logic.

        Returns:
            True if the job succeeded, False if all retries were exhausted.
        """
        if job_id not in self._jobs:
            raise ValueError(f"Job ID {job_id} not found in the queue.")

        job_state = self._jobs[job_id]
        data = job_state["data"]
        
        attempt = 0
        while attempt <= self.MAX_RETRIES:
            try:
                print(f"\n--- Attempt {attempt + 1} for Job {job_id} ---")
                # Execute the job processor
                processor(data)
                
                # Success
                print(f"Job {job_id} succeeded on attempt {attempt + 1}.")
                return True

            except Exception as e:
                print(f"Job {job_id} failed on attempt {attempt + 1}. Error: {e}")
                
                # Check if we have exhausted all retries
                if attempt >= self.MAX_RETRIES:
                    print(f"Job {job_id} failed permanently after {self.MAX_RETRIES + 1} attempts.")
                    return False

                # Prepare for retry
                delay = self._calculate_backoff_delay(attempt)
                job_state["retries"] += 1
                attempt += 1
                
                # Simulate backoff delay
                print(f"Retrying job {job_id} in {delay} seconds...")
                # In a real system, we would sleep here: time.sleep(delay)
                time.sleep(0.01) # Use minimal sleep for testing speed
        
        return False # Should be unreachable if logic is correct, but serves as a safeguard

# --- Example Usage ---

# 1. Define a processor that fails a set number of times
FAILURE_COUNT = 2
attempt_counter = 0

def flaky_fetch_url(data: Dict[str, Any]):
    """
    A mock processor that fails the first two times and succeeds on the third.
    """
    global attempt_counter
    attempt_counter += 1
    
    if attempt_counter <= FAILURE_COUNT:
        raise ConnectionError(f"Simulated network failure (Attempt {attempt_counter})")
    
    print(f"Successfully fetched URL: {data['url']}")
    # Simulate successful work
    return {"status": "success"}

# 2. Define a processor that fails every time
always_fail_processor = lambda data: 1 / 0

# 3. Initialize Queue
queue = JobQueue()

# --- Scenario 1: Successful Retry ---
print("===================================================")
print("SCENARIO 1: Job succeeds after retries (Expected: True)")
queue.add_job("job_success", {"url": "https://example.com"})
attempt_counter = 0 # Reset counter for the first job
success = queue.process_job("job_success", flaky_fetch_url)
print(f"Final result for job_success: {success}")


# --- Scenario 2: Failure Exhaustion ---
print("\n===================================================")
print("SCENARIO 2: Job fails all retries (Expected: False)")
queue.add_job("job_failure", {"data": "critical"})
attempt_counter = 0 # Reset counter for the second job
failure = queue.process_job("job_failure", always_fail_processor)
print(f"Final result for job_failure: {failure}")