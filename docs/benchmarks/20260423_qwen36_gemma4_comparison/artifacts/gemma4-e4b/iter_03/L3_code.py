from collections import deque
from typing import Optional, Tuple, Dict

class JobQueue:
    """
    Implements a priority-based job scheduling queue.
    Higher priority numbers (2=Critical) are processed first.
    FIFO is maintained for jobs of the same priority.
    """
    def __init__(self):
        # Using a list of deques, indexed by priority level (0, 1, 2).
        # Max priority is 2.
        self.queue: list[deque] = [deque() for _ in range(3)]

    def add_job(self, job_name: str, job_data: Dict, priority: int = 0) -> None:
        """
        Adds a job to the queue at the specified priority level.
        Priority must be between 0 and 2.
        """
        if not 0 <= priority <= 2:
            raise ValueError("Priority must be 0, 1, or 2.")
        
        # Jobs are appended to the end of the deque, maintaining FIFO order for that priority level.
        self.queue[priority].append((job_name, job_data))

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """
        Retrieves and removes the highest priority job available.
        Returns (job_name, job_data) or None if the queue is empty.
        """
        # Iterate priorities from highest (2) down to lowest (0)
        for priority in range(2, -1, -1):
            if self.queue[priority]:
                # Pop the job from the front (FIFO) of the highest available priority queue
                return self.queue[priority].popleft()
        
        return None

    def is_empty(self) -> bool:
        """Checks if the queue contains any jobs."""
        return all(not q for q in self.queue)

if __name__ == '__main__':
    queue = JobQueue()

    # Test Case 1: Mixed priorities
    print("--- Test Case 1: Mixed Priorities ---")
    queue.add_job("normal_a", {"type": "normal"}, priority=0)
    queue.add_job("critical_a", {"type": "critical"}, priority=2)
    queue.add_job("high_a", {"type": "high"}, priority=1)
    queue.add_job("normal_b", {"type": "normal"}, priority=0)
    queue.add_job("critical_b", {"type": "critical"}, priority=2)

    # Expected order: Critical_a, Critical_b, High_a, Normal_a, Normal_b
    
    jobs = []
    while not queue.is_empty():
        job = queue.get_next_job()
        if job:
            jobs.append(f"{job[0]} (P={2 if job[0] == 'critical_a' else 2 if job[0] == 'critical_b' else 1 if job[0] == 'high_a' else 0})")

    print("Processing Order:", jobs)
    # Output should reflect priority 2 first, then 1, then 0 (FIFO within groups)
    
    print("\n--- Test Case 2: FIFO within same priority ---")
    queue2 = JobQueue()
    queue2.add_job("fifo_1", {}, priority=1)
    queue2.add_job("fifo_2", {}, priority=1)
    queue2.add_job("fifo_3", {}, priority=1)

    jobs2 = []
    while not queue2.is_empty():
        job = queue2.get_next_job()
        jobs2.append(job[0])
    
    print("FIFO Order:", jobs2) # Expected: ['fifo_1', 'fifo_2', 'fifo_3']