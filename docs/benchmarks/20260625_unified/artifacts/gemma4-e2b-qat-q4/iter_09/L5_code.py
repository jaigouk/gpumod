from dataclasses import dataclass
from typing import Callable, Any, Dict, Tuple, List
from collections import defaultdict

@dataclass
class Job:
    id: str
    data: Dict[str, Any]
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: Dict[str, Any]) -> Tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                # Attempt the function call
                result = fn(data)
                # If successful, return immediately
                return True, attempts + 1
            except Exception as e:
                attempts += 1
                # In a real scenario, backoff logic would go here.
                # Per requirement, we just retry immediately.
                continue

        # If loop finishes without success
        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        # Stores jobs grouped by priority: {priority: deque([job1, job2, ...])}
        # Using deque ensures FIFO order within the same priority level.
        self.jobs: Dict[int, List[Job]] = defaultdict(list)
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: Dict[str, Any], priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self.jobs[priority].append(job)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        # 1. Find the corresponding job data
        target_job = None
        for priority in sorted(self.jobs.keys(), reverse=True):
            for job in self.jobs[priority]:
                if job.id == job_id:
                    target_job = job
                    break
            if target_job:
                break

        if not target_job:
            raise ValueError(f"Job with ID {job_id} not found.")

        data = target_job.data

        # 2. Use RetryPolicy to execute the processor
        success, attempts_made = self.retry_policy.run(processor, data)

        # Update job metadata (optional, but good practice)
        target_job.retries = attempts_made - 1
        return success

    def get_next_job(self) -> Tuple[str, Dict[str, Any]] | None:
        """Returns the highest-priority job's (id, data); FIFO order within same priority."""
        if not self.jobs:
            return None

        # Find the highest existing priority level
        highest_priority = max(self.jobs.keys())

        # Return the next job from the FIFO queue of that highest priority
        if self.jobs[highest_priority]:
            next_job = self.jobs[highest_priority].pop(0)
            return (next_job.id, next_job.data)

        # Should not be reached if checks above are correct, but safety return
        return None

if __name__ == '__main__':
    # Example Usage
    queue = JobQueue()

    # Define a test processor function
    def risky_operation(data: dict) -> bool:
        print(f"Processing data: {data}. Attempting operation...")
        # Simulate failure twice, then success
        global attempt_count
        attempt_count += 1

        if attempt_count <= 2:
            print(f"  --> Attempt {attempt_count}: Failed.")
            raise ConnectionError("Simulated network failure.")
        else:
            print(f"  --> Attempt {attempt_count}: Success.")
            return True

    attempt_count = 0

    # Add jobs
    queue.add_job("j1", {"task": "low_p"}, priority=1)
    queue.add_job("j2", {"task": "high_p"}, priority=10)
    queue.add_job("j3", {"task": "med_p"}, priority=5)

    print("--- Initial State ---")
    print(f"Next Job: {queue.get_next_job()}") # j2 (Priority 10)

    # Process j2 (should succeed on 3rd attempt)
    print("\n--- Processing Job j2 ---")
    success_j2 = queue.process_job("j2", risky_operation)
    print(f"Job j2 processing result: {success_j2}")
    print(f"Job j2 retries remaining: {queue.jobs[10][0].retries}")

    print("\n--- State after processing j2 ---")
    print(f"Next Job: {queue.get_next_job()}") # j3 (Priority 5)

    # Process j3
    print("\n--- Processing Job j3 ---")
    success_j3 = queue.process_job("j3", risky_operation)
    print(f"Job j3 processing result: {success_j3}")