import time
from typing import Callable, Dict, Any, Optional

class JobQueue:
    """
    A job queue implementation with retry logic and exponential backoff.
    """
    MAX_RETRIES = 3
    BACKOFF_DELAYS = [1, 2, 4]  # Seconds for 1st, 2nd, 3rd retry (after initial failure)

    def __init__(self):
        # Stores job data: {job_id: job_data}
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        """Adds a job to the queue."""
        self.jobs[job_id] = {
            "data": data,
            "retries": 0
        }

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        """
        Processes a job with retry logic and exponential backoff.

        Args:
            job_id: The ID of the job to process.
            processor: The function (processor) that executes the job logic.

        Returns:
            True if the job succeeded, False otherwise.
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job ID '{job_id}' not found in the queue.")

        job_info = self.jobs[job_id]
        max_attempts = self.MAX_RETRIES + 1  # Initial attempt + 3 retries

        for attempt in range(max_attempts):
            try:
                # Attempt execution
                processor(job_info["data"])
                
                # Success
                return True
            
            except Exception as e:
                print(f"--- Job {job_id} failed on attempt {attempt + 1}. Error: {e}")
                
                # Check if we have exhausted all retries
                if attempt == self.MAX_RETRIES:
                    print(f"--- Job {job_id} failed permanently after {self.MAX_RETRIES} retries.")
                    return False

                # Calculate backoff delay
                retry_count = attempt + 1
                delay = self.BACKOFF_DELAYS[retry_count - 1]
                
                job_info["retries"] += 1
                
                # Simulate backoff delay
                print(f"--- Retrying job {job_id} in {delay} seconds (Attempt {attempt + 2}/{max_attempts}).")
                # In a real system, we would sleep(delay) here.
                # Since the requirement is to track rather than actually sleep:
                time.sleep(0.001) # Minimal sleep for simulation visibility
        
        return False # Should theoretically be unreachable if logic is correct

# ======================================================================
# Example Usage and Testing
# ======================================================================

# 1. Setup
queue = JobQueue()

# --- Test Case 1: Successful Job ---
def successful_processor(data):
    print(f"\n[Success Test] Processing job with data: {data}")
    return "OK"

queue.add_job("job_success", {"url": "http://good.com"})

# --- Test Case 2: Job that fails and eventually succeeds (Simulated) ---
# This function will fail the first two times, and succeed on the third.
failure_counter = 0
def intermittent_processor(data):
    global failure_counter
    failure_counter += 1
    print(f"\n[Intermittent Test] Attempt {failure_counter}...")
    if failure_counter <= 2:
        raise ConnectionError("Network timeout simulated.")
    return "Success after recovery"

queue.add_job("job_intermittent", {"url": "http://recovery.com"})

# --- Test Case 3: Job that fails permanently ---
permanent_failure_counter = 0
def permanently_failing_processor(data):
    global permanent_failure_counter
    permanent_failure_counter += 1
    print(f"\n[Failure Test] Attempt {permanent_failure_counter}...")
    raise PermissionError("Access denied permanently.")

queue.add_job("job_fail_permanent", {"url": "http://bad.com"})


# 2. Execution

print("\n=======================================================")
print("STARTING JOB: job_success")
success_result = queue.process_job("job_success", successful_processor)
print(f"Final Result for job_success: {success_result}")

print("\n=======================================================")
print("STARTING JOB: job_intermittent (Should succeed on 3rd attempt)")
intermittent_result = queue.process_job("job_intermittent", intermittent_processor)
print(f"Final Result for job_intermittent: {intermittent_result}")

print("\n=======================================================")
print("STARTING JOB: job_fail_permanent (Should fail after 3 retries)")
permanent_result = queue.process_job("job_fail_permanent", permanently_failing_processor)
print(f"Final Result for job_fail_permanent: {permanent_result}")