from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Tuple, Optional

@dataclass
class Job:
    id: str
    data: Dict[str, Any]
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable[..., Any], data: Dict[str, Any]) -> Tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                result = fn(data)
                return True, attempts + 1
            except Exception as e:
                attempts += 1
                # Log or handle exception if necessary, but the requirement is not to sleep
        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        # Stores jobs grouped by priority level. 
        # priority_levels[p] will be a list of jobs with priority p.
        self.priority_levels: Dict[int, List[Job]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any], priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority)
        if priority not in self.priority_levels:
            self.priority_levels[priority] = []
        self.priority_levels[priority].append(job)

    def process_job(self, job_id: str, processor: Callable[..., Any]) -> bool:
        """
        Pulls the highest priority job and attempts to process it using RetryPolicy.
        Returns True if successful, False otherwise.
        """
        if not self.priority_levels:
            return False

        # 1. Get the highest priority key
        highest_priority = max(self.priority_levels.keys())

        # 2. Get the FIFO job from that priority level
        job_list = self.priority_levels[highest_priority]
        if not job_list:
            return False

        current_job = job_list.pop(0)

        print(f"Processing job {job_id} (Priority: {current_job.priority})...")

        # 3. Run with RetryPolicy
        success, attempts_made = RetryPolicy(max_attempts=3).run(processor, current_job.data)

        if not success:
            # Re-queue failed job (simple mechanism: just add it back to its original priority level)
            current_job.retries += 1
            self.priority_levels[current_job.priority].append(current_job)
            print(f"Job {job_id} failed after {attempts_made} attempts. Retrying (Attempt {current_job.retries}).")
            return False

        return True

    def get_next_job(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Returns the highest priority job's (id, data). 
        FIFO order within the same priority.
        """
        if not self.priority_levels:
            return None

        # Find the highest priority key
        highest_priority = max(self.priority_levels.keys())

        # Get the FIFO job from that priority level
        job_list = self.priority_levels[highest_priority]
        if not job_list:
            return None

        job = job_list[0]

        # Return the required tuple (id, data)
        return (job.id, job.data)

if __name__ == "__main__":
    # --- Example Usage ---

    def sample_processor(data: Dict[str, Any]):
        # Simulate an operation that fails twice
        if data.get("fail_count", 0) < 2:
            data["fail_count"] = data.get("fail_count", 0) + 1
            raise ValueError("Transient error occurred")

        data["fail_count"] = data.get("fail_count", 0) + 1
        print(f"SUCCESS: Job {data['id']} processed successfully.")
        return f"Processed: {data['id']}"

    queue = JobQueue()

    # Add jobs
    queue.add_job("J1", {"name": "Low P Job"}, priority=10)
    queue.add_job("J2", {"name": "High P Job"}, priority=1)
    queue.add_job("J3", {"name": "Mid P Job"}, priority=5)
    queue.add_job("J4", {"name": "High P Job 2"}, priority=1)

    # Demonstrate get_next_job
    print("\n--- Next Job Check ---")
    job1 = queue.get_next_job()
    print(f"Next job: {job1}") # Should be J2 (priority 1)

    # Demonstrate processing (J2 should succeed quickly)
    print("\n--- Processing J2 ---")
    queue.process_job("J2", sample_processor)

    # Add a new high priority job
    queue.add_job("J5", {"name": "Critical Job"}, priority=100)

    # Demonstrate highest priority job handling (J5)
    print("\n--- Next Job Check 2 ---")
    job2 = queue.get_next_job()
    print(f"Next job: {job2}") # Should be J5 (priority 100)

    print("\n--- Processing J5 ---")
    queue.process_job("J5", sample_processor)

    # Demonstrate a job that requires retries (J1 starts at priority 10)
    queue.add_job("J6", {"name": "Retry Job"}, priority=10)
    print("\n--- Processing J1 ---")
    # J1 will fail twice, then succeed on the 3rd attempt (because max_attempts=3 is default in process_job logic)
    queue.process_job("J1", sample_processor)

    # Final check
    print(f"\nQueue empty check: {queue.get_next_job()}")