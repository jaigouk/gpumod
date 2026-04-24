import heapq
from typing import Optional, Tuple, Dict

class JobQueue:
    """
    A priority-based queue for scheduling jobs.
    Higher priority numbers (2 > 1 > 0) are processed first.
    FIFO order is maintained for jobs of the same priority.
    """
    def __init__(self):
        # The heap stores tuples: (-priority, counter, job_name, job_data)
        # Using negative priority converts the min-heap into a max-priority structure.
        self.queue: list[tuple[int, int, str, dict]] = []
        # Counter ensures FIFO order for jobs with identical priority
        self.counter = 0

    def add_job(self, job_name: str, job_data: Dict, priority: int = 0) -> None:
        """
        Adds a job to the queue with a specified priority.
        Priority levels: 0 (Normal), 1 (High), 2 (Critical).
        """
        # We use negative priority because heapq is a min-heap, 
        # and we want the highest priority number (2) to be treated as the smallest key (-2).
        priority_key = -priority
        
        entry = (priority_key, self.counter, job_name, job_data)
        heapq.heappush(self.queue, entry)
        self.counter += 1

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """
        Retrieves and removes the highest priority job from the queue.
        Returns a tuple (job_name, job_data) or None if the queue is empty.
        """
        if not self.queue:
            return None
        
        # Pop the smallest item (which corresponds to the highest priority)
        priority_key, _, job_name, job_data = heapq.heappop(self.queue)
        
        return job_name, job_data

if __name__ == '__main__':
    queue = JobQueue()
    
    # Test Case 1: Standard Priority Mixing
    print("--- Test Case 1: Priority Mixing ---")
    queue.add_job("normal_a", {"type": "normal"}, priority=0)
    queue.add_job("critical_a", {"type": "critical"}, priority=2)
    queue.add_job("high_a", {"type": "high"}, priority=1)
    queue.add_job("normal_b", {"type": "normal"}, priority=0)

    print("Processing jobs:")
    while True:
        job = queue.get_next_job()
        if job is None:
            break
        name, data = job
        print(f"  -> Processed: {name} (Type: {data['type']})")
    # Expected order: critical_a, high_a, normal_a, normal_b (FIFO for normal)

    print("\n--- Test Case 2: FIFO Tie Breaker ---")
    queue2 = JobQueue()
    
    # Add three jobs with the same priority (0)
    queue2.add_job("fifo_1", {"type": "normal"}, priority=0)
    queue2.add_job("fifo_2", {"type": "normal"}, priority=0)
    queue2.add_job("fifo_3", {"type": "normal"}, priority=0)

    print("Processing FIFO jobs:")
    while True:
        job = queue2.get_next_job()
        if job is None:
            break
        name, data = job
        print(f"  -> Processed: {name}")
    # Expected order: fifo_1, fifo_2, fifo_3

    print("\n--- Test Case 3: Empty Queue ---")
    queue3 = JobQueue()
    job = queue3.get_next_job()
    print(f"Attempting to get job from empty queue: {job}")