import heapq
from typing import Optional, Tuple, Dict

class JobQueue:
    """
    Priority-based job scheduling queue.
    Higher priority numbers (2=Critical) are processed first.
    Jobs of the same priority maintain FIFO order.
    """
    def __init__(self):
        # The heap stores tuples: (-priority, sequence_number, job_name, job_data)
        # Negating priority turns the min-heap into a max-heap for priority.
        self.queue = []
        self.sequence_counter = 0

    def add_job(self, job_name: str, job_data: Dict, priority: int = 0) -> None:
        """
        Adds a job to the queue with a specified priority.
        Priority levels: 0=Normal, 1=High, 2=Critical.
        """
        # Use the sequence counter to ensure FIFO for jobs with the same priority
        entry = (-priority, self.sequence_counter, job_name, job_data)
        heapq.heappush(self.queue, entry)
        self.sequence_counter += 1

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """
        Retrieves and removes the highest priority job from the queue.
        Returns (job_name, job_data) or None if the queue is empty.
        """
        if not self.queue:
            return None
        
        # Pop the smallest item (which corresponds to the highest priority)
        _, _, job_name, job_data = heapq.heappop(self.queue)
        return (job_name, job_data)

if __name__ == '__main__':
    # Example 1: Basic Priority Test
    print("--- Test 1: Priority Order ---")
    queue1 = JobQueue()
    queue1.add_job("normal_1", {"type": "normal"}, priority=0)
    queue1.add_job("critical_1", {"type": "critical"}, priority=2)
    queue1.add_job("high_1", {"type": "high"}, priority=1)

    job1 = queue1.get_next_job()
    print(f"Next job: {job1[0]} (Expected: critical_1)")

    job2 = queue1.get_next_job()
    print(f"Next job: {job2[0]} (Expected: high_1)")

    job3 = queue1.get_next_job()
    print(f"Next job: {job3[0]} (Expected: normal_1)")

    # Example 2: FIFO Tie-breaker Test
    print("\n--- Test 2: FIFO Tie-breaker ---")
    queue2 = JobQueue()
    queue2.add_job("high_A", {"type": "high"}, priority=1) # Added first
    queue2.add_job("normal_A", {"type": "normal"}, priority=0)
    queue2.add_job("high_B", {"type": "high"}, priority=1) # Added second

    job_a = queue2.get_next_job()
    print(f"Next job: {job_a[0]} (Expected: high_A)") # high_A should come before high_B

    job_b = queue2.get_next_job()
    print(f"Next job: {job_b[0]} (Expected: high_B)")

    job_c = queue2.get_next_job()
    print(f"Next job: {job_c[0]} (Expected: normal_A)")

    # Example 3: Empty Queue
    print("\n--- Test 3: Empty Queue ---")
    queue3 = JobQueue()
    job_empty = queue3.get_next_job()
    print(f"Next job: {job_empty} (Expected: None)")