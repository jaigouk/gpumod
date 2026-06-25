from dataclasses import dataclass
from typing import Callable, Any, List, Tuple
import heapq

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable[..., Any], data: Any) -> Tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                result = fn(data)
                # If the function returns a truthy value, we consider it successful
                if result:
                    return True, attempts
                else:
                    # If the function executes but returns None or False, we treat it as needing retry
                    attempts += 1
                    continue
            except Exception as e:
                attempts += 1
                continue
        return False, attempts

class JobQueue:
    def __init__(self, retry_policy: RetryPolicy):
        self._jobs: List[Job] = []
        self._policy = retry_policy
        self._insertion_counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        new_job = Job(id=job_id, data=data, priority=priority)
        # Store jobs in a format suitable for the priority queue (heapq)
        # We use a tuple: (-priority, insertion_order, job_id)
        # Negative priority ensures higher priority comes first.
        # Insertion order ensures FIFO for jobs with the same priority.
        heapq.heappush(self._jobs, (-new_job.priority, self._insertion_counter, new_job))
        self._insertion_counter += 1

    def process_job(self, job_id: str, processor: Callable[..., Any]) -> bool:
        """
        Uses RetryPolicy to process a job.
        The processor function passed here is what RetryPolicy calls.
        """
        # Find the job to process (simple linear search for ID matching)
        target_job = None

        # Temporarily move all items to a list to sort easily, 
        # then pop to find the job or remove it.
        temp_storage = []

        while self._jobs:
            neg_p, insertion_id, job = heapq.heappop(self._jobs)
            if job.id == job_id:
                target_job = job
            else:
                temp_storage.append((neg_p, insertion_id, job))

        # Put back non-target jobs
        for item in temp_storage:
            heapq.heappush(self._jobs, item)

        if not target_job:
            return False # Job not found

        # Execute the job using the RetryPolicy
        success, attempts = self._policy.run(processor, target_job.data)

        if success:
            print(f"Job {job_id} succeeded after {attempts} attempts.")
            # Job successfully completed and removed from queue
            return True
        else:
            print(f"Job {job_id} failed after {attempts} attempts.")
            # Job failed and remains in the queue (or could be moved to a dead-letter queue)
            return False

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest-priority job's (id, data); FIFO order within the same priority.
        """
        if not self._jobs:
            return None

        # Peek at the top item (highest priority) without removing it
        # The structure is: (-priority, insertion_order, Job object)
        neg_p, _, job = self._jobs[0]
        return (job.id, job.data)

# Example Usage (not part of the final required solution, but used for verification):
if __name__ == '__main__':
    # Define a mock processor that simulates failure occasionally
    failure_counter = 0
    def mock_processor(data):
        global failure_counter
        if failure_counter < 3:
            failure_counter += 1
            raise ConnectionError("Simulated temporary network failure")
        return True # Success

    # 1. Setup
    policy = RetryPolicy(max_attempts=3)
    queue = JobQueue(policy)

    # 2. Add jobs
    queue.add_job("job1", {"task": "Critical"}, priority=10)
    queue.add_job("job2", {"task": "Normal"}, priority=1)
    queue.add_job("job3", {"task": "Critical_1"}, priority=10)
    queue.add_job("job4", {"task": "Normal_2"}, priority=1)

    # 3. Get next job (Highest priority: job1 or job3, FIFO)
    print("Next job:", queue.get_next_job())

    # 4. Process job1 (Will retry twice and succeed on 3rd attempt if mock fails 2 times)
    print("\n--- Processing job1 ---")
    success = queue.process_job("job1", mock_processor)
    print(f"Job 1 Processing Success: {success}")

    # 5. Get next job (Next highest priority is job3)
    print("\nNext job:", queue.get_next_job())

    # 6. Process job3
    print("\n--- Processing job3 ---")
    success = queue.process_job("job3", mock_processor)
    print(f"Job 3 Processing Success: {success}")

    # 7. Get next job (Next priority is job2 or job4, FIFO)
    print("\nNext job:", queue.get_next_job())

    # 8. Process job2
    print("\n--- Processing job2 ---")
    success = queue.process_job("job2", mock_processor)
    print(f"Job 2 Processing Success: {success}")