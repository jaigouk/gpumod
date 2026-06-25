from dataclasses import dataclass
from typing import Callable, Any, tuple
import heapq

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    """Encapsulates retry-with-backoff logic."""
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: Any) -> tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                result = fn(data)
                return True, attempts + 1
            except Exception as e:
                attempts += 1
                # In a real scenario, a backoff sleep would occur here.
                # We return failure immediately upon the final attempt if the loop finishes.
        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        # Stores jobs as: [(-priority, sequence_number, job_id, data)]
        # Using negative priority simulates max-priority ordering in a min-heap.
        # sequence_number ensures FIFO within the same priority level.
        self._queue = []
        self._job_counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._job_counter += 1
        sequence_number = self._job_counter
        job = Job(id=job_id, data=data, priority=priority)

        # Push to the heap: (-priority, sequence, job_id, data)
        heapq.heappush(self._queue, (-priority, sequence_number, job_id, data))

    def _pop_next(self) -> tuple[str, dict] | None:
        """Retrieves the highest priority job using FIFO tie-breaker."""
        if not self._queue:
            return None

        neg_priority, seq, job_id, job_data = heapq.heappop(self._queue)
        return job_id, job_data

    def get_next_job(self) -> tuple[str, dict] | None:
        """Returns the highest-priority job's (id, data)."""
        return self._pop_next()

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Executes a job using the RetryPolicy.
        The processor function takes the job data and returns a result.
        """
        # Step 1: Retrieve the job data
        job_data = self.get_next_job()

        if job_data is None:
            return False

        # Step 2: Define the function to be executed with retries
        def job_fn():
            return processor(job_data)

        # Step 3: Apply the RetryPolicy
        policy = RetryPolicy(max_attempts=5)

        success, attempts_made = policy.run(job_fn, job_data)

        if success:
            # If successful, we consumed the job and processed it
            return True

        # If not successful, we might want to re-add the job if the retry policy
        # logic was external, but based on the prompt structure, the job is considered
        # failed/processed if max attempts are reached. 
        # Since we used get_next_job(), the job is removed upon call.
        return False


if __name__ == '__main__':
    # Example Usage:

    queue = JobQueue()

    def failing_processor(data):
        print(f"Attempting to process job {data['id']} (Priority: {data['priority']})...")
        # Simulate failure on the first 2 attempts, success on the 3rd
        if data['id'] == 'JobA':
            raise RuntimeError("Temporary failure.")
        if data['id'] == 'JobB':
            raise ValueError("Fatal error.")

        return f"Job {data['id']} successfully completed!"

    def successful_processor(data):
        return f"Job {data['id']} processed normally."

    # Add jobs
    queue.add_job('JobA', {'id': 'JobA', 'priority': 10}, priority=10) # High priority
    queue.add_job('JobC', {'id': 'JobC', 'priority': 5}, priority=5)  # Medium priority
    queue.add_job('JobB', {'id': 'JobB', 'priority': 5}, priority=5)  # Medium priority (FIFO with C)
    queue.add_job('JobD', {'id': 'JobD', 'priority': 10}, priority=10) # Highest priority

    print("--- Processing Job D (Highest Priority) ---")
    success_d = queue.process_job('JobD', successful_processor)
    print(f"Job D result: {'Success' if success_d else 'Failure'}\n")

    print("--- Processing Job A (Needs Retries) ---")
    success_a = queue.process_job('JobA', successful_processor)
    print(f"Job A result: {'Success' if success_a else 'Failure'}\n")

    print("--- Processing Job C (Medium Priority, FIFO with Job B) ---")
    # Job B was added after Job C
    success_c = queue.process_job('JobC', successful_processor)
    print(f"Job C result: {'Success' if success_c else 'Failure'}\n")

    print("--- Processing Job B (Medium Priority, FIFO) ---")
    success_b = queue.process_job('JobB', successful_processor)
    print(f"Job B result: {'Success' if success_b else 'Failure'}\n")