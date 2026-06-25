import collections
from typing import Dict, Any, Tuple, Optional

class JobQueue:
    """
    Implements a priority-based job scheduling queue.
    Priorities: 2 (Critical) > 1 (High) > 0 (Normal)
    """
    def __init__(self):
        # Map priority level to a deque (FIFO queue for that priority)
        # We initialize with all defined levels for easy access
        self.queues: Dict[int, collections.deque] = {
            0: collections.deque(),  # Normal
            1: collections.deque(),  # High
            2: collections.deque()   # Critical
        }

    def add_job(self, name: str, payload: Dict[str, Any], priority: int = 0):
        """
        Adds a job to the appropriate priority queue.
        """
        if priority not in self.queues:
            raise ValueError(f"Invalid priority level: {priority}. Must be 0, 1, or 2.")

        self.queues[priority].append((name, payload))

    def get_next_job(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Returns the highest priority job available. 
        If multiple jobs share the highest priority, returns the oldest one (FIFO).
        Returns None if the queue is empty.
        """
        # Iterate through priorities in descending order (2, 1, 0)
        for priority in [2, 1, 0]:
            queue = self.queues[priority]
            if queue:
                name, payload = queue.popleft()
                return (name, payload)

        return None

if __name__ == '__main__':
    queue = JobQueue()
    queue.add_job("normal", {"type": "normal"}, priority=0)
    queue.add_job("critical", {"type": "critical"}, priority=2)
    queue.add_job("high", {"type": "high"}, priority=1)
    queue.add_job("normal_2", {"type": "normal_2"}, priority=0)

    # Test 1: Critical job (priority 2) first
    job1 = queue.get_next_job()
    print(f"Job 1: {job1}") # Expected: ('critical', {'type': 'critical'})

    # Test 2: High priority job (priority 1) next
    job2 = queue.get_next_job()
    print(f"Job 2: {job2}") # Expected: ('high', {'type': 'high'})

    # Test 3: Normal priority jobs (priority 0) in FIFO order
    job3 = queue.get_next_job()
    print(f"Job 3: {job3}") # Expected: ('normal', {'type': 'normal'})

    job4 = queue.get_next_job()
    print(f"Job 4: {job4}") # Expected: ('normal_2', {'type': 'normal_2'})

    # Test 4: Queue empty
    job5 = queue.get_next_job()
    print(f"Job 5: {job5}") # Expected: None