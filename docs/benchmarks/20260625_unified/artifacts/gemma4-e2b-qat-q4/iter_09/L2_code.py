from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        # Stores job data: {job_id: {"data": dict, "attempts": int}}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        # Stores metadata for tracking: {job_id: {"attempts": int, "backoff_delays": list[int]}}
        self.job_metadata: Dict[str, Dict[str, Any]] = {}
        self.MAX_ATTEMPTS = 4

    def add_job(self, job_id: str, data: dict):
        """Adds a job to the queue."""
        if job_id not in self.jobs:
            self.jobs[job_id] = {"data": data, "attempts": 0}
            self.job_metadata[job_id] = {
                "attempts": 0,
                "backoff_delays": []
            }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Processes a job with retries and exponential backoff.

        Returns True on success, False if all attempts fail.
        Returns the calculated delays if retries occur.
        """
        if job_id not in self.jobs:
            raise KeyError(f"Job ID {job_id} not found in queue.")

        job_data = self.jobs[job_id]["data"]
        meta = self.job_metadata[job_id]

        # Attempt loop (Initial attempt + MAX_ATTEMPTS - 1 retries)
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            if attempt > 1:
                # If this is not the first attempt, execute the backoff delay
                delay = meta["backoff_delays"][attempt - 2]
                print(f"Job {job_id}: Retrying after {delay}s (Attempt {attempt})...")

            try:
                processor(job_data)
                return True

            except Exception as e:
                print(f"Job {job_id}: Attempt {attempt} failed. Error: {e}")

                if attempt < self.MAX_ATTEMPTS:
                    # Record the exponential backoff delay
                    # Backoff sequence: 1s, 2s, 4s
                    if attempt == 2:
                        backoff_time = 1
                    elif attempt == 3:
                        backoff_time = 2
                    else: # attempt == 4 (the last attempt, no further delay needed if it fails)
                        backoff_time = 4

                    meta["backoff_delays"].append(backoff_time)
                    meta["attempts"] = attempt

                # If it's the last attempt (attempt == 4), we just let the exception propagate 
                # after recording the final delay, but the loop will exit and return False

        # If the loop completes without returning True, all attempts failed
        return False

if __name__ == '__main__':
    # --- Example Usage ---

    def failing_processor(data):
        """A processor that fails the first 3 times, succeeds on the 4th."""
        if not hasattr(failing_processor, 'call_count'):
            failing_processor.call_count = 0
        failing_processor.call_count += 1
        if failing_processor.call_count < 4:
            raise RuntimeError(f"Simulated failure on call {failing_processor.call_count}")
        return True

    def always_failing_processor(data):
        """A processor that always fails."""
        raise ValueError("Critical service down.")

    # Test 1: Success on 4th attempt
    queue = JobQueue()
    JOB_ID_1 = "job_A"
    queue.add_job(JOB_ID_1, {"payload": "test_success"})

    result_1 = queue.process_job(JOB_ID_1, failing_processor)
    print("\n--- Test 1 Result ---")
    print(f"Job {JOB_ID_1} Success: {result_1}")
    print(f"Job {JOB_ID_1} Final Metadata: {queue.job_metadata[JOB_ID_1]}")
    print("-" * 20)


    # Test 2: Complete failure
    JOB_ID_2 = "job_B"
    queue.add_job(JOB_ID_2, {"payload": "test_failure"})

    result_2 = queue.process_job(JOB_ID_2, always_failing_processor)
    print("\n--- Test 2 Result ---")
    print(f"Job {JOB_ID_2} Success: {result_2}")
    print(f"Job {JOB_ID_2} Final Metadata: {queue.job_metadata[JOB_ID_2]}")
    print("-" * 20)