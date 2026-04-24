import heapq
from typing import Optional, Tuple, Dict

class JobQueue:
    """
    A priority-based job queue using a min-heap implementation.
    Higher priority values are processed first. FIFO order is maintained
    for jobs with the same priority.
    """
    def __init__(self):
        # The heap stores tuples: (priority_key, sequence_number, job_name, job_data)
        # priority_key is negative priority to simulate a max-heap using a min-heap.
        self.queue = []
        self.counter = 0

    def add_job(self, job_name: str, job_data: Dict, priority: int = 0) -> None:
        """
        Adds a job to the queue with a specified priority.
        Priority levels: 0 (Normal), 1 (High), 2 (Critical).
        """
        # Use negative priority so that the highest priority (e.g., 2 -> -2)
        # appears as the smallest element in the min-heap.
        priority_key = -priority
        
        # The counter ensures FIFO order for jobs with the same priority.
        heapq.heappush(self.queue, (priority_key, self.counter, job_name, job_data))
        self.counter += 1

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """
        Retrieves and removes the highest priority job from the queue.
        Returns a tuple (job_name, job_data) or None if the queue is empty.
        """
        if not self.queue:
            return None
        
        # Pop the item with the smallest priority_key (highest actual priority)
        # and the smallest sequence_number (oldest job).
        _, _, job_name, job_data = heapq.heappop(self.queue)
        return (job_name, job_data)

if __name__ == '__main__':
    # --- Example 1: Standard Priority Check ---
    print("--- Example 1: Priority Check ---")
    queue1 = JobQueue()
    
    # Normal (0)
    queue1.add_job("job_normal_1", {"type": "normal"}, priority=0) 
    # Critical (2)
    queue1.add_job("job_critical_1", {"type": "critical"}, priority=2)
    # High (1)
    queue1.add_job("job_high_1", {"type": "high"}, priority=1) 
    # Critical (2) - second one
    queue1.add_job("job_critical_2", {"type": "critical"}, priority=2)

    # Expected order: Critical 1, Critical 2, High 1, Normal 1
    
    job = queue1.get_next_job()
    print(f"Next job 1: {job[0]} (Expected: job_critical_1)")

    job = queue1.get_next_job()
    print(f"Next job 2: {job[0]} (Expected: job_critical_2)")

    job = queue1.get_next_job()
    print(f"Next job 3: {job[0]} (Expected: job_high_1)")

    job = queue1.get_next_job()
    print(f"Next job 4: {job[0]} (Expected: job_normal_1)")

    job = queue1.get_next_job()
    print(f"Next job 5: {job} (Expected: None)")

    # --- Example 2: FIFO Tie-Breaker Check ---
    print("\n--- Example 2: FIFO Tie-Breaker ---")
    queue2 = JobQueue()
    
    # Both are priority 1 (High). job_A was added first.
    queue2.add_job("job_A", {"type": "high"}, priority=1) 
    # Both are priority 1 (High). job_B was added second.
    queue2.add_job("job_B", {"type": "high"}, priority=1) 
    
    # Critical job inserted in the middle
    queue2.add_job("job_C_crit", {"type": "critical"}, priority=2)

    # Expected order: job_C_crit, job_A, job_B
    
    job = queue2.get_next_job()
    print(f"Next job 1: {job[0]} (Expected: job_C_crit)")

    job = queue2.get_next_job()
    print(f"Next job 2: {job[0]} (Expected: job_A)")

    job = queue2.get_next_job()
    print(f"Next job 3: {job[0]} (Expected: job_B)")