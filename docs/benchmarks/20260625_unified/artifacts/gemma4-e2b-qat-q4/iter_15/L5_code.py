from dataclasses import dataclass
from typing import Callable, Any, List, Optional, Tuple
import time

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable[..., Any], data: dict) -> Tuple[bool, int]:
        attempts_made = 0

        while attempts_made < self.max_attempts:
            try:
                result = fn(data)
                # Assuming success if fn returns without exception
                return True, attempts_made + 1
            except Exception:
                attempts_made += 1
                # If this was the last attempt, break to return False
                if attempts_made == self.max_attempts:
                    return False, attempts_made
        return False, attempts_made

class JobQueue:
    def __init__(self):
        self._jobs: List[Job] = []

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        new_job = Job(id=job_id, data=data, priority=priority)
        self._jobs.append(new_job)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Retrieves a job, applies the RetryPolicy, and removes it if successful.
        """
        # 1. Find the job to process
        current_jobs = self._jobs
        job_to_process_index = -1

        for i, job in enumerate(current_jobs):
            if job.id == job_id:
                job_to_process_index = i
                break

        if job_to_process_index == -1:
            return False # Job not found

        job = current_jobs.pop(job_to_process_index)

        # 2. Apply RetryPolicy
        policy = RetryPolicy()
        success, attempts_made = policy.run(processor, job.data)

        if success:
            # Update job metadata if necessary (optional, but good practice)
            job.retries = attempts_made - 1
            return True
        else:
            # Return job to queue if it failed all attempts
            self._jobs.append(job)
            return False

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        """
        Returns the highest-priority job's (id, data); FIFO order within the same priority.
        """
        if not self._jobs:
            return None

        # Sort criteria:
        # 1. Priority (descending: highest priority first)
        # 2. Insertion order (FIFO: earlier jobs first)

        # We sort by: (-priority, insertion_index). 
        # Since we don't explicitly track insertion index, we can use the actual list index 
        # before sorting if we process the list directly, but using the index 
        # within a stable sort (like Timsort in Python) guarantees FIFO for equal keys.

        indexed_jobs = list(enumerate(self._jobs))

        sorted_jobs = sorted(indexed_jobs, key=lambda x: (-x[1].priority, x[0]))

        # Get the first job after sorting
        index, job = sorted_jobs[0]
        return (job.id, job.data)

if __name__ == '__main__':
    # Example Usage:

    def mock_processor(data: dict) -> str:
        """Simulates a processor that fails twice and succeeds on the third attempt."""
        if not hasattr(mock_processor, 'call_count'):
            mock_processor.call_count = 0

        mock_processor.call_count += 1
        print(f"  [Processor] Attempt {mock_processor.call_count} processing data: {data['value']}")

        if mock_processor.call_count < 3:
            raise ValueError("Transient failure.")

        return f"Processing successful for {data['value']}"

    queue = JobQueue()

    # Add jobs: High priority (10) first, then Low priority (1)
    queue.add_job("J1", {"value": "TaskA"}, priority=10)
    queue.add_job("J2", {"value": "TaskB"}, priority=1)
    queue.add_job("J3", {"value": "TaskC"}, priority=5)

    print("--- Initial State ---")
    print(f"Next Job: {queue.get_next_job()}")

    # Process J1 (Should succeed on 3rd try)
    print("\n--- Processing J1 ---")
    success1 = queue.process_job("J1", mock_processor)
    print(f"J1 Success: {success1}")

    print("\n--- State after J1 ---")
    print(f"Next Job: {queue.get_next_job()}")

    # Process J2 (Should succeed on 1st try)
    print("\n--- Processing J2 ---")
    success2 = queue.process_job("J2", mock_processor)
    print(f"J2 Success: {success2}")

    print("\n--- Processing J3 ---")
    success3 = queue.process_job("J3", mock_processor)
    print(f"J3 Success: {success3}")

    print("\n--- Final State ---")
    print(f"Next Job: {queue.get_next_job()}")

    # Testing failure scenario (Job that always fails)
    def always_fail_processor(data: dict) -> str:
        raise RuntimeError("Permanent error.")

    always_fail_job = JobQueue()
    always_fail_job.add_job("J4", {"value": "FailingTask"}, priority=10)

    print("\n--- Testing Always Fail Job J4 ---")
    # RetryPolicy max_attempts = 4. It should fail 4 times and be re-queued.
    success4 = always_fail_job.process_job("J4", always_fail_processor)
    print(f"J4 Success: {success4}")
    print(f"Next Job after J4 attempt: {always_fail_job.get_next_job()}")