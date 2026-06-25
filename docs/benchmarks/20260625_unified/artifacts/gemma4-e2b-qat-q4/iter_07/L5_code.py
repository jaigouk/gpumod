from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: Any) -> tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                return True, attempts
            except Exception:
                attempts += 1
        return False, attempts

class JobQueue:
    def __init__(self):
        # Stores Job objects, keyed by job_id for quick access
        self._jobs: dict[str, Job] = {}

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        policy = RetryPolicy(max_attempts=3)  # Default policy

        success, attempts_made = policy.run(processor, job.data)

        if success:
            # Update retries count if successful
            self._jobs[job_id] = job._replace(retries=job.retries + 1)
            return True

        # If failed, update retries count
        self._jobs[job_id] = job._replace(retries=job.retries + 1)
        return False

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._jobs:
            return None

        # Find the job with the highest priority.
        # Tie-breaking: FIFO order (implicitly achieved by iterating over items 
        # in insertion order if we keep a list structure, but here we sort explicitly).

        best_job_id = None
        highest_priority = -float('inf')

        # Sort jobs: Primary key is priority (desc), secondary key is original insertion order (implicit FIFO).
        # Since we only care about the highest priority, we iterate to find the maximum.

        for job_id, job in self._jobs.items():
            if job.priority > highest_priority:
                highest_priority = job.priority
                best_job_id = job_id
            elif job.priority == highest_priority:
                # If priorities are equal, the one encountered first in the iteration 
                # (which depends on dict insertion order in modern Python) 
                # is chosen, satisfying FIFO for that priority level.
                pass

        if best_job_id:
            job = self._jobs[best_job_id]
            return (job.id, job.data)

        return None

if __name__ == '__main__':
    # Example usage demonstration
    q = JobQueue()

    def task(data):
        print(f"Processing {data}")

    q.add_job("j1", {"message": "task 1", "priority": 1})
    q.add_job("j2", {"message": "task 2", "priority": 5})
    q.add_job("j3", {"message": "task 3", "priority": 1})
    q.add_job("j4", {"message": "task 4", "priority": 1})

    print("Next job:", q.get_next_job()) # j2 (priority 5)

    # Simulate j2 processing (success)
    print("\nProcessing j2 (Success):", q.process_job("j2", task))
    print("Next job:", q.get_next_job()) # j1, j3, j4 (priority 1, FIFO tie break)

    # Simulate failure for j1 (RetryPolicy max attempts is 3)
    def failing_task(data):
        print(f"Attempt: {data['message']}. Failing...")
        if data['retries'] < 2:
            raise ValueError("Transient error")
        print(f"Attempt: {data['message']} (Retry {data['retries']}). Success!")
        return "Success"

    print("\nProcessing j1 (Failure/Retry):", q.process_job("j1", failing_task))
    print("Job j1 retries:", q._jobs["j1"].retries)

    print("Next job:", q.get_next_job()) # j3 (priority 1, FIFO)