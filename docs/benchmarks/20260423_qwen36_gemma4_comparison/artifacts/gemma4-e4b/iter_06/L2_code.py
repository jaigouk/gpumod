import time
from typing import Dict, Any, Callable

class JobQueue:
    """
    A job queue implementation with built-in retry logic and exponential backoff.
    """
    def __init__(self):
        # Stores job details: {job_id: {"data": Any, "retries": 0}}
        self._jobs: Dict[str, Dict[str, Any]] = {}
        # Maximum number of retries allowed (initial attempt + 3 retries = 4 total attempts)
        self.MAX_RETRIES = 3

    def add_job(self, job_id: str, data: Dict[str, Any]):
        """Adds a new job to the queue."""
        if job_id in self._jobs:
            raise ValueError(f"Job ID {job_id} already exists.")
        self._jobs[job_id] = {"data": data, "retries": 0}

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Processes a job with retry logic and exponential backoff.

        Args:
            job_id: The ID of the job to process.
            processor: The function that performs the job logic.

        Returns:
            True if the job succeeded, False if all retries were exhausted.
        """
        if job_id not in self._jobs:
            raise ValueError(f"Job ID {job_id} not found in the queue.")

        job_data = self._jobs[job_id]["data"]
        
        # The loop runs for the initial attempt + MAX_RETRIES
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                # Attempt to process the job
                processor(job_data)
                
                # Success
                return True
            
            except Exception as e:
                # Job failed
                print(f"Job {job_id} failed on attempt {attempt + 1}. Error: {e}")

                # Check if we have exhausted all retries
                if attempt >= self.MAX_RETRIES:
                    print(f"Job {job_id} failed permanently after {self.MAX_RETRIES + 1} attempts.")
                    return False

                # Calculate exponential backoff delay (1s, 2s, 4s, ...)
                # Delay = 2^attempt (where attempt is 0, 1, 2...)
                delay = 2 ** attempt
                
                # Update retry count (optional, but good practice)
                self._jobs[job_id]["retries"] = attempt + 1
                
                # Simulate the backoff wait
                print(f"Retrying job {job_id} in {delay} seconds...")
                # In a real scenario, we would use time.sleep(delay)
                # We skip actual sleeping as per requirements.
                pass 

        return False # Should be unreachable if logic is correct, but serves as a final fallback

# --- Example Usage ---

# Mock processor that fails a specific number of times
class MockProcessor:
    def __init__(self, fail_count: int):
        self.fail_count = fail_count
        self.attempts = 0

    def __call__(self, data: Dict[str, Any]):
        self.attempts += 1
        print(f"--- Processing job {data['job_id']} (Attempt {self.attempts}) ---")
        if self.attempts <= self.fail_count:
            raise ConnectionError("Simulated Network Failure")
        print(f"--- Job {data['job_id']} succeeded! ---")
        return True

# 1. Successful Job (Fails 2 times, succeeds on 3rd)
print("="*50)
print("TEST CASE 1: Job succeeds after retries")
print("="*50)
queue = JobQueue()
job1_data = {"job_id": "job1"}
queue.add_job("job1", job1_data)

# This processor will fail 2 times, succeeding on the 3rd attempt (Attempt 3 <= MAX_RETRIES + 1)
processor_success = MockProcessor(fail_count=2) 
success = queue.process_job("job1", processor_success)
print(f"\n[RESULT] Job 1 success status: {success}")


# 2. Failed Job (Fails 4 times, exhausting retries)
print("\n" + "="*50)
print("TEST CASE 2: Job fails permanently")
print("="*50)
queue_fail = JobQueue()
job2_data = {"job_id": "job2"}
queue_fail.add_job("job2", job2_data)

# This processor will fail 4 times (Attempt 4 > MAX_RETRIES + 1), exhausting retries
processor_fail = MockProcessor(fail_count=4) 
success_fail = queue_fail.process_job("job2", processor_fail)
print(f"\n[RESULT] Job 2 success status: {success_fail}")