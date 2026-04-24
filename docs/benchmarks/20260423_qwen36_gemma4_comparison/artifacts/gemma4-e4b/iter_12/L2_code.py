import time
from typing import Callable, Dict, Any, Optional

class JobQueue:
    """
    A job queue implementation with retry logic and exponential backoff.
    """
    def __init__(self):
        # Stores job data: {job_id: job_data}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        # Stores internal state for tracking retries: {job_id: retry_count}
        self.retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        """
        Adds a new job to the queue.
        """
        if job_id in self.jobs:
            raise ValueError(f"Job ID {job_id} already exists.")
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Processes a job with retry logic and exponential backoff.

        Args:
            job_id: The ID of the job to process.
            processor: The function that attempts to execute the job.

        Returns:
            True if the job succeeded, False if all retries were exhausted.
        """
        if job_id not in self.jobs:
            raise KeyError(f"Job ID {job_id} not found in the queue.")

        MAX_RETRIES = 3
        
        while self.retry_counts[job_id] <= MAX_RETRIES:
            try:
                # Attempt to process the job
                processor(self.jobs[job_id])
                
                # Success
                print(f"Job {job_id} succeeded on attempt {self.retry_counts[job_id] + 1}.")
                return True

            except Exception as e:
                self.retry_counts[job_id] += 1
                
                if self.retry_counts[job_id] > MAX_RETRIES:
                    print(f"Job {job_id} failed after {MAX_RETRIES + 1} attempts. Max retries exhausted. Error: {e}")
                    return False

                # Calculate backoff delay: 2^n - 1 (1s, 3s, 7s... or simpler 2^n)
                # Using 2^n where n is the current attempt count (starting from 0 for first retry)
                # Attempt 1 (Retry 1): 2^0 = 1s
                # Attempt 2 (Retry 2): 2^1 = 2s
                # Attempt 3 (Retry 3): 2^2 = 4s
                delay = 2 ** (self.retry_counts[job_id] - 1)
                
                print(f"Job {job_id} failed on attempt {self.retry_counts[job_id]}. Retrying in {delay} seconds...")
                
                # Simulate delay
                time.sleep(delay)
        
        return False


# --- Example Usage ---

# 1. Setup a mock failure processor
call_count = 0

def failing_fetch_url(data):
    """A processor that fails the first two times but succeeds on the third."""
    global call_count
    call_count += 1
    print(f"--- Attempt {call_count} for {data.get('job_id')} ---")
    
    if call_count < 3:
        raise ConnectionError("Network timeout or service unavailable.")
    
    print(f"--- Success! Data processed: {data} ---")
    return True

# 2. Setup a permanent failure processor
def always_failing_processor(data):
    raise ValueError("Critical permanent error.")

# 3. Initialize Queue
queue = JobQueue()

# --- Test Case 1: Success after retries ---
job_id_success = "job1"
queue.add_job(job_id_success, {"job_id": job_id_success, "url": "https://example.com"})

print("\n=======================================")
print("TEST CASE 1: Success after 2 retries (3 attempts total)")
print("=======================================")
success_result = queue.process_job(job_id_success, failing_fetch_url)
print(f"\nFinal result for {job_id_success}: {'SUCCESS' if success_result else 'FAILURE'}")

# Reset state for the next test
call_count = 0
queue = JobQueue()

# --- Test Case 2: Exhausted retries ---
job_id_fail = "job2"
queue.add_job(job_id_fail, {"job_id": job_id_fail, "url": "https://badurl.com"})

print("\n=======================================")
print("TEST CASE 2: Failure after 3 retries (4 attempts total)")
print("=======================================")
failure_result = queue.process_job(job_id_fail, always_failing_processor)
print(f"\nFinal result for {job_id_fail}: {'SUCCESS' if failure_result else 'FAILURE'}")