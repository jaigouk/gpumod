from dataclasses import dataclass
from typing import Callable, Any, List, Tuple, Optional

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    """
    Encapsulates retry-with-backoff logic. Does not actually sleep.
    """
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: Any) -> Tuple[bool, int]:
        """
        Executes fn(data), retries on exception until success or max_attempts reached.

        :param fn: The function to execute.
        :param data: The input data for the function.
        :return: A tuple (success: bool, attempts_made: int).
        """
        attempts = 0
        while attempts < self.max_attempts:
            try:
                result = fn(data)
                return True, attempts + 1
            except Exception:
                attempts += 1
        return False, attempts

class JobQueue:
    """
    Orchestrates jobs and manages processing using a RetryPolicy.
    """
    def __init__(self):
        # Stores (priority, insertion_order, job_id, data)
        self._queue: List[Tuple[int, int, str, dict]] = []
        self._insertion_counter = 0
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        """Adds a new job to the queue."""
        self._insertion_counter += 1
        # We store priority first for easy sorting and handling
        self._queue.append((priority, self._insertion_counter, job_id, data))
        # Keep the queue sorted by priority (highest priority first)
        self._queue.sort(key=lambda x: x[0], reverse=True)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Retrieves the highest priority job and executes the processor using RetryPolicy.

        :param job_id: The ID of the job to process.
        :param processor: The callable function to execute on the data.
        :return: True if job succeeded, False otherwise.
        """
        # 1. Find the target job in the queue
        target = None
        target_index = -1

        # Find the first job matching job_id (FIFO within the same priority)
        for i, (_, _, j_id, _) in enumerate(self._queue):
            if j_id == job_id:
                target = self._queue[i]
                target_index = i
                break

        if target is None:
            # Job not found in queue
            return False

        priority, insertion_order, _, job_data = target

        # Remove the job from the active queue
        self._queue.pop(target_index)

        # 2. Run the job using the RetryPolicy
        # The processor function takes the job_data and returns some result (we ignore it for boolean check)
        success, attempts_made = self.retry_policy.run(processor, job_data)

        return success

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        """
        Returns the highest-priority job's (id, data); FIFO order within the same priority.
        Returns None if the queue is empty.
        """
        if not self._queue:
            return None

        # The queue is already sorted by priority.
        # We take the first element, which is the highest priority job.
        priority, insertion_order, job_id, job_data = self._queue[0]

        return (job_id, job_data)

if __name__ == '__main__':
    # --- Example Usage ---

    # Define a function that simulates failure 2 times then success
    execution_count = 0
    def demo_processor(data):
        global execution_count
        execution_count += 1
        print(f"Executing job {data['id']} (Attempt {execution_count}). Data: {data}")
        if execution_count < 3:
            raise RuntimeError("Simulated transient error")
        return f"Processed {data['id']}"

    # 1. Initialize Queue and Policy
    queue = JobQueue()

    # 2. Add Jobs
    queue.add_job("job_A", {"task": "critical", "val": 10}, priority=10)
    queue.add_job("job_B", {"task": "normal", "val": 1}, priority=1)
    queue.add_job("job_C", {"task": "high_prio_1", "val": 5}, priority=5)
    queue.add_job("job_D", {"task": "normal", "val": 2}, priority=1) # Same priority as B, should come after B (FIFO)
    queue.add_job("job_E", {"task": "critical", "val": 20}, priority=10) # Same priority as A, should come after A

    print("--- Adding Jobs ---")
    print(f"Next job: {queue.get_next_job()}") # job_A (Priority 10, Insertion 1)

    print("\n--- Processing Jobs ---")

    # Process Job A (Critical, should pass retry on attempt 1 or 2)
    success_a = queue.process_job("job_A", demo_processor)
    print(f"\nResult for Job A processing: {'Success' if success_a else 'Failed'}")

    # Process Job C (Priority 5)
    success_c = queue.process_job("job_C", demo_processor)
    print(f"Result for Job C processing: {'Success' if success_c else 'Failed'}")

    # Process Job B (Priority 1, FIFO first)
    success_b = queue.process_job("job_B", demo_processor)
    print(f"Result for Job B processing: {'Success' if success_b else 'Failed'}")

    # Process Job D (Priority 1, FIFO second)
    success_d = queue.process_job("job_D", demo_processor)
    print(f"Result for Job D processing: {'Success' if success_d else 'Failed'}")

    # Process Job E (Priority 10, FIFO second)
    success_e = queue.process_job("job_E", demo_processor)
    print(f"Result for Job E processing: {'Success' if success_e else 'Failed'}")

    print("\n--- Queue Empty Check ---")
    print(f"Next job: {queue.get_next_job()}")