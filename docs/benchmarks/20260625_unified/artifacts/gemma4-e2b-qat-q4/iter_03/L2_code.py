from typing import Callable, Dict, Any, Tuple

class JobQueue:
    def __init__(self):
        # Stores job_id: {'data': data, 'retries': count}
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = {
            "data": data,
            "retries": 0
        }

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        if job_id not in self.jobs:
            # Assuming job_id must exist if we want to process it.
            return False

        current_job = self.jobs[job_id]
        data = current_job["data"]
        retries = current_job["retries"]

        max_attempts = 4

        for attempt in range(1, max_attempts + 1):
            try:
                result = processor(data)

                # Success: Return True and update job state if this is the final attempt
                if attempt == 1:
                    # On first success, update the record (if success is final)
                    current_job["retries"] = 0

                return True

            except Exception as e:
                # Exception occurred

                if attempt < max_attempts:
                    # Record the delay required for the next attempt (Exponential backoff: 2^(attempt-1))
                    delay = 2**(attempt - 1)

                    # Record delay as data (logging)
                    print(f"Job {job_id} failed on attempt {attempt}. Retrying in {delay}s. Exception: {e}")

                    # Increment retry count for tracking
                    current_job["retries"] += 1

                    # The loop continues automatically for the next attempt
                else:
                    # Last attempt failed
                    print(f"Job {job_id} failed permanently after {max_attempts} attempts.")
                    # Return False
                    return False

        # Should be unreachable if logic above is complete, but required for safety
        return False

# Example usage (optional, for testing)
if __name__ == '__main__':
    q = JobQueue()

    def failing_processor(data: Dict[str, Any]) -> None:
        # Fails on the first three calls, succeeds on the fourth
        if not hasattr(failing_processor, 'count'):
            failing_processor.count = 0
        failing_processor.count += 1
        if failing_processor.count < 4:
            raise ValueError("Transient error")
        print("Processor succeeded!")

    # Test Case 1: Success within 4 attempts
    job_id_1 = "job_A"
    q.add_job(job_id_1, {"payload": "test_1"})
    success1 = q.process_job(job_id_1, failing_processor)
    print(f"\nResult for job {job_id_1}: {success1}") # Expected: True

    # Reset job state for next test
    q.jobs["job_B"] = {"data": {"payload": "test_2"}, "retries": 0}

    # Test Case 2: Permanent failure
    def always_failing_processor(data: Dict[str, Any]) -> None:
        raise RuntimeError("Fatal error")

    job_id_2 = "job_B"
    q.add_job(job_id_2, {"payload": "test_2"})
    success2 = q.process_job(job_id_2, always_failing_processor)
    print(f"\nResult for job {job_id_2}: {success2}") # Expected: False