from dataclasses import dataclass
from typing import Callable, Any, Optional, Tuple

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: dict) -> Tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                # Attempt the function
                result = fn(data)
                return True, attempts + 1
            except Exception as e:
                attempts += 1
                # In a real scenario, backoff would happen here.
                # We just increment the attempt count and retry.
                pass
        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        self._jobs = []

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(job_id=job_id, data=data, priority=priority)
        self._jobs.append(job)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = next((j for j in self._jobs if j.id == job_id), None)

        if job is None:
            return False

        policy = RetryPolicy()
        success, attempts_made = policy.run(processor, job.data)

        # Update job metadata if successful (optional but good practice)
        if success:
            job.retries = attempts_made - 1
            self._jobs.remove(job)
            return True
        return False

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if not self._jobs:
            return None

        # Find the highest priority job (highest integer priority value)
        # For FIFO within the same priority, we look for the max value and then
        # the index of the first occurrence of that value.

        best_job: Optional[Job] = None
        max_priority = -float('inf')

        # Iterate to find the best job (highest priority, FIFO)
        for job in self._jobs:
            if job.priority > max_priority:
                max_priority = job.priority
                best_job = job
            # If priority is equal, we keep the existing best_job (FIFO relative to the list order)

        if best_job:
            return (best_job.id, best_job.data)
        return None

if __name__ == '__main__':
    # Example Usage Demonstration

    # Mock processing function that fails twice then succeeds
    call_count = 0
    def mock_processor(data):
        global call_count
        call_count += 1
        print(f"--- Attempt {call_count}: Processing job for data: {data}")
        if call_count <= 2:
            raise ValueError("Temporary failure")
        return f"Success for {data}"

    queue = JobQueue()
    queue.add_job("Job_A", {"task": "Task 1", "value": 1}, priority=10)
    queue.add_job("Job_B", {"task": "Task 2", "value": 2}, priority=5)
    queue.add_job("Job_C", {"task": "Task 3", "value": 3}, priority=10)

    print("\n--- Processing Job B (Priority 5) ---")
    # Job B should succeed on the first attempt (since it only fails twice, and this is the first run)
    # Wait, if mock_processor fails twice, the policy runs up to 4 attempts.
    # Attempt 1 (fails), Attempt 2 (fails), Attempt 3 (succeeds). Attempts made = 3.
    success = queue.process_job("Job_B", mock_processor)
    print(f"Job B processed successfully: {success}")


    print("\n--- Processing Job A (Priority 10) ---")
    # Job A should succeed on the first attempt if Job B is removed
    success_a = queue.process_job("Job_A", mock_processor)
    print(f"Job A processed successfully: {success_a}")

    print("\n--- Processing Job C (Priority 10) ---")
    # Job C should succeed on the first attempt
    success_c = queue.process_job("Job_C", mock_processor)
    print(f"Job C processed successfully: {success_c}")

    print("\n--- Getting Next Job ---")
    next_job = queue.get_next_job()
    print(f"Next job retrieved: {next_job}")