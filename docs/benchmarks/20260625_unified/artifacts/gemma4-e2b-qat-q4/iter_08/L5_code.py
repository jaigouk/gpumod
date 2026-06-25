from dataclasses import dataclass
from typing import Callable, Any, Optional, Tuple, List

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
            attempts += 1
            try:
                return fn(data)
            except Exception:
                # In a real scenario, a backoff sleep would happen here.
                # Per requirement: Do NOT actually sleep.
                if attempts == self.max_attempts:
                    break
                continue
        return False, attempts

class JobQueue:
    def __init__(self):
        self._jobs: List[Job] = []

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        new_job = Job(id=job_id, data=data, priority=priority)
        self._jobs.append(new_job)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        # Find the job by ID to process
        job_to_process = None
        for job in self._jobs:
            if job.id == job_id:
                job_to_process = job
                break

        if job_to_process is None:
            return False

        # Use RetryPolicy to execute the processor
        retry_policy = RetryPolicy(max_attempts=4)
        success, attempts = retry_policy.run(processor, job_to_process.data)

        # Update job metadata if needed (optional, but good practice)
        job_to_process.retries = attempts

        return success

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if not self._jobs:
            return None

        # Sort jobs: Primary sort by priority (desc), secondary sort by insertion order (FIFO)
        # We sort by (-priority, original index) to handle priority descending, 
        # and stable sort for FIFO.

        # Create temporary sortable list with index for stable FIFO
        indexed_jobs = []
        for i, job in enumerate(self._jobs):
            indexed_jobs.append((job.priority, i, job))

        # Sort: highest priority first (descending priority), then FIFO (ascending index i)
        indexed_jobs.sort(key=lambda x: (-x[0], x[1]))

        # Return the ID and data of the first job
        _, _, next_job = indexed_jobs[0]
        return (next_job.id, next_job.data)

if __name__ == "__main__":
    # Example usage demonstration (not required, but helpful for testing logic)
    q = JobQueue()

    # Add jobs
    q.add_job("job1", {"task": "low_p"}, priority=1)
    q.add_job("job2", {"task": "high_p"}, priority=10)
    q.add_job("job3", {"task": "med_p"}, priority=5)

    # Process a job (requires a job with a processor function)
    def successful_processor(data):
        print(f"Processing successful job: {data['task']}")
        return True

    print(f"Processing highest priority job: {q.get_next_job()}") # Job2

    success = q.process_job("job2", successful_processor)
    print(f"Job 2 processed successfully: {success}")
    print(f"Job 2 retries: {q._jobs[1].retries}\n")


    # Demonstrate retry logic (failing processor)
    def failing_processor(data):
        print(f"Attempting to process failing job: {data['task']}")
        raise ValueError("Transient error")

    # Add a job specifically designed to fail
    q.add_job("job_fail", {"task": "critical"}, priority=100)

    # Process job with retry policy (will fail 4 times, then succeed/fail)
    success_fail = q.process_job("job_fail", failing_processor)
    print(f"Job fail processed successfully: {success_fail}")
    print(f"Job fail retries: {q._jobs[3].retries}")
    print(f"Job fail final status: {q._jobs[3].retries < 4}")