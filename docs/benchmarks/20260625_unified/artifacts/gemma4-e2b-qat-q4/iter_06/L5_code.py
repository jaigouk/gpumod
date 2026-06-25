from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, Optional, List

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
        attempt = 0
        while attempt < self.max_attempts:
            try:
                result = fn(data)
                return True, attempt + 1  # Success
            except Exception as e:
                attempt += 1
                # No actual sleep is performed, as per requirements

        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        # Store jobs. {job_id: Job object}
        self._jobs: Dict[str, Job] = {}
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        self._jobs[job_id] = job

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]

        # Use the RetryPolicy to execute the processor
        success, attempts = self.retry_policy.run(processor, job.data)

        if success:
            # Optionally, update the job's retry count if needed
            job.retries = attempts - 1
            return True
        else:
            # Optional: handle persistent failure, e.g., logging or re-queuing outside this specific function
            job.retries = attempts
            return False

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if not self._jobs:
            return None

        # Sort jobs: Primary key is priority (descending), secondary key is insertion order (implicitly handled by sorted list)
        sorted_jobs: List[Job] = list(self._jobs.values())

        # Sort by priority (descending) and then by insertion order (stable sort)
        sorted_jobs.sort(key=lambda j: (j.priority, -j.id))

        # Find the job that is next
        next_job = sorted_jobs[0]
        return (next_job.id, next_job.data)

if __name__ == '__main__':
    # Example Usage

    # 1. Setup the Queue
    queue = JobQueue()

    # 2. Add Jobs
    queue.add_job("job_a", {"payload": "short", "task": "short"}, priority=1)
    queue.add_job("job_b", {"payload": "long_fail"}, priority=5)
    queue.add_job("job_c", {"payload": "medium"}, priority=3)

    # 3. Define a processor function (simulating failure)
    count_a = 0
    def process_a(data):
        nonlocal count_a
        count_a += 1
        print(f"Executing job_a. Attempt {count_a}. Data: {data}")
        if count_a < 3:
            raise ValueError("Temporary failure")
        return {"status": "completed", "job_id": "job_a"}

    # 4. Test Processing (Job A with retries)
    print("\n--- Processing Job A (will retry) ---")
    success = queue.process_job("job_a", process_a)
    print(f"Job A success: {success}")
    print(f"Job A retries: {queue._jobs['job_a'].retries}")


    # 5. Test Processing (Job B with high priority)
    def process_b(data):
        print(f"Executing job_b. Data: {data}")
        return {"status": "completed", "job_id": "job_b"}

    print("\n--- Processing Job B (high priority) ---")
    success = queue.process_job("job_b", process_b)
    print(f"Job B success: {success}")

    # 6. Test Getting Next Job (Priority should be 5 then 3 then 1)
    print("\n--- Getting Next Jobs ---")

    job1 = queue.get_next_job()
    if job1:
        print(f"Next Job 1 (Priority 5): {job1[0]} (Priority check)")

    job2 = queue.get_next_job()
    if job2:
        print(f"Next Job 2 (Priority 3): {job2[0]} (Priority check)")

    job3 = queue.get_next_job()
    if job3:
        print(f"Next Job 3 (Priority 1): {job3[0]} (Priority check)")