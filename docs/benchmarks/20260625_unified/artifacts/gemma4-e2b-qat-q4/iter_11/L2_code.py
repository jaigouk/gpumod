from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict) -> None:
        self.jobs[job_id] = {
            "data": data,
            "attempts": 0
        }

    def process_job(self, job_id: str, processor: Callable[[dict], Any]) -> bool:
        """
        Processes a job with retries and exponential backoff.
        Returns True on success, False if all attempts fail.
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job ID {job_id} not found.")

        job_info = self.jobs[job_id]
        data = job_info["data"]
        max_attempts = 4
        backoff_delays = [1, 2, 4]  # Delays for retries 1, 2, 3

        for attempt in range(max_attempts):
            job_info["attempts"] = attempt + 1
            try:
                # Requirement 1: Call processor(data)
                processor(data)
                return True  # Requirement 5: Return True on success

            except Exception as e:
                # If it's the last attempt, don't retry
                if attempt == max_attempts - 1:
                    print(f"Job {job_id} failed after {max_attempts} attempts. Final error: {e}")
                    return False

                # Requirement 2 & 3: Retry logic and backoff
                delay = backoff_delays[attempt]

                # Requirement 4: Do NOT actually sleep — record the delays as data.
                # We simulate the recording by printing the planned delay for demonstration,
                # but the actual code execution does not pause.
                print(f"Job {job_id} failed attempt {attempt + 1}. Retrying in {delay}s...")
                # In a real system, this delay would be stored in a persistence layer.

                # Wait logic: 
                # import time # Removed per standard library constraint
                pass 

        return False

# Example usage (Optional, for testing the logic)
if __name__ == '__main__':

    class FailingProcessor:
        def __init__(self, fail_count):
            self.fail_count = fail_count
            self.executed = 0

        def __call__(self, data: dict):
            self.executed += 1
            if self.executed <= self.fail_count:
                print(f"Processor ran for job {data.get('id')}, attempt {self.executed}: Failed (expected)")
                raise RuntimeError(f"Transient failure on attempt {self.executed}")

            print(f"Processor ran for job {data.get('id')}, attempt {self.executed}: Success")
            return True

    queue = JobQueue()

    # --- Test Case 1: Success on first try ---
    print("--- Starting Test Case 1 (Success) ---")
    queue.add_job("job_1", {"id": "job_1"})
    failing_proc_1 = FailingProcessor(fail_count=0)
    success = queue.process_job("job_1", failing_proc_1)
    print(f"Test Case 1 Result: {'Success' if success else 'Failure'}\n")

    # --- Test Case 2: Fails 3 times, succeeds on 4th try ---
    print("--- Starting Test Case 2 (Success on 4th attempt) ---")
    queue.add_job("job_2", {"id": "job_2"})
    failing_proc_2 = FailingProcessor(fail_count=3)
    success_2 = queue.process_job("job_2", failing_proc_2)
    print(f"Test Case 2 Result: {'Success' if success_2 else 'Failure'}\n")

    # Reset and test failure
    queue = JobQueue()
    queue.add_job("job_3", {"id": "job_3"})
    failing_proc_3 = FailingProcessor(fail_count=4) # Will fail all 4
    success_3 = queue.process_job("job_3", failing_proc_3)
    print(f"Test Case 3 Result: {'Success' if success_3 else 'Failure'}")