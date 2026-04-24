import time
from typing import Dict, Any, Callable, Optional

class JobQueue:
    """
    Manages job processing with retry logic and exponential backoff.
    """
    def __init__(self):
        # Stores job data: {job_id: job_data}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        # Stores job metadata (like retry count): {job_id: {'retries': int}}
        self.job_metadata: Dict[str, Dict[str, int]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        """Adds a new job to the queue."""
        if job_id in self.jobs:
            raise ValueError(f"Job ID {job_id} already exists.")
        self.jobs[job_id] = data
        self.job_metadata[job_id] = {'retries': 0}
        print(f"Job {job_id} added successfully.")

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Processes a job with retry logic and exponential backoff.

        Args:
            job_id: The ID of the job to process.
            processor: The function that executes the job logic.

        Returns:
            True if the job succeeded, False if all retries were exhausted.
        """
        if job_id not in self.jobs:
            print(f"Error: Job ID {job_id} not found.")
            return False

        job_data = self.jobs[job_id]
        max_retries = 3
        
        for attempt in range(max_retries + 1):
            try:
                print(f"\n--- Attempt {attempt + 1} for Job {job_id} ---")
                
                # Execute the processor function with job data
                processor(job_data)
                
                print(f"SUCCESS: Job {job_id} completed successfully on attempt {attempt + 1}.")
                return True

            except Exception as e:
                current_retries = self.job_metadata[job_id]['retries']
                
                print(f"FAILURE: Job {job_id} failed on attempt {attempt + 1}. Error: {e}")

                if current_retries < max_retries:
                    # Calculate exponential backoff delay: 2^current_retries seconds
                    # Attempt 1 (0 retries) -> Wait 1s (2^0)
                    # Attempt 2 (1 retry) -> Wait 2s (2^1)
                    # Attempt 3 (2 retries) -> Wait 4s (2^2)
                    delay = 2 ** current_retries
                    
                    self.job_metadata[job_id]['retries'] += 1
                    
                    print(f"Retrying Job {job_id} in {delay} second(s)...")
                    # Simulate sleeping
                    # time.sleep(delay) 
                    pass 
                else:
                    print(f"CRITICAL: Job {job_id} failed after {max_retries + 1} attempts. Exhausted retries.")
                    return False
        
        return False # Should technically be unreachable if loop logic is correct

# --- Example Usage ---

# 1. Setup: Mock processor that fails a specific number of times
class MockProcessor:
    def __init__(self, fail_count: int):
        self.fail_count = fail_count
        self.calls = 0

    def __call__(self, data: Dict[str, Any]):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise ConnectionError(f"Simulated network failure on call {self.calls}")
        print(f"Processor executed successfully (Call {self.calls}).")

# 2. Initialization
queue = JobQueue()

# --- Scenario 1: Success after retries (Fails 2 times, succeeds on 3rd) ---
print("==========================================================")
print("SCENARIO 1: Job succeeds after retries (2 failures)")
print("==========================================================")
processor_success = MockProcessor(fail_count=2)
queue.add_job("job_success", {"url": "https://api.example.com"})

success_job = queue.process_job("job_success", processor_success)
print(f"\n[Result] Job 'job_success' final status: {'SUCCESS' if success_job else 'FAILED'}")


# --- Scenario 2: Failure after all retries (Fails 4 times) ---
print("\n\n==========================================================")
print("SCENARIO 2: Job fails after exhausting retries (4 failures)")
print("==========================================================")
processor_failure = MockProcessor(fail_count=4)
queue.add_job("job_failure", {"url": "https://api.fail.com"})

failure_job = queue.process_job("job_failure", processor_failure)
print(f"\n[Result] Job 'job_failure' final status: {'SUCCESS' if failure_job else 'FAILED'}")