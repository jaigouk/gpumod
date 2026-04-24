from collections import deque
from typing import Optional, Tuple, Dict, Any

class JobQueue:
    """
    Implements a priority-based job queue.
    Higher priority numbers (2 > 1 > 0) indicate higher urgency.
    Jobs of the same priority maintain FIFO order.
    """
    def __init__(self):
        # Stores queues, keyed by priority level (0, 1, 2).
        # Each value is a deque (FIFO queue).
        self.queues: Dict[int, deque] = {0: deque(), 1: deque(), 2: deque()}

    def add_job(self, job_name: str, job_data: Dict[str, Any], priority: int = 0) -> None:
        """
        Adds a job to the queue with the specified priority.
        """
        if priority not in self.queues:
            raise ValueError(f"Invalid priority level: {priority}. Must be 0, 1, or 2.")
        
        job = (job_name, job_data)
        self.queues[priority].append(job)

    def get_next_job(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Retrieves and removes the highest priority job.
        Priorities are checked in descending order (2 -> 1 -> 0).
        """
        # Iterate through priorities from highest (2) to lowest (0)
        for priority in sorted(self.queues.keys(), reverse=True):
            if self.queues[priority]:
                # Found a job at this priority level
                job_name, job_data = self.queues[priority].popleft()
                return job_name, job_data
        
        # If all queues are empty
        return None

    def is_empty(self) -> bool:
        """
        Checks if the queue is empty.
        """
        return all(not q for q in self.queues.values())

if __name__ == '__main__':
    queue = JobQueue()

    # Test case setup
    queue.add_job("normal_1", {"type": "normal"}, priority=0)
    queue.add_job("critical_1", {"type": "critical"}, priority=2)
    queue.add_job("normal_2", {"type": "normal"}, priority=0)
    queue.add_job("high_1", {"type": "high"}, priority=1)
    queue.add_job("critical_2", {"type": "critical"}, priority=2)
    queue.add_job("high_2", {"type": "high"}, priority=1)

    print("--- Processing Jobs ---")
    
    # Expected order:
    # 1. critical_1 (Priority 2, FIFO)
    # 2. critical_2 (Priority 2, FIFO)
    # 3. high_1 (Priority 1, FIFO)
    # 4. high_2 (Priority 1, FIFO)
    # 5. normal_1 (Priority 0, FIFO)
    # 6. normal_2 (Priority 0, FIFO)

    while not queue.is_empty():
        job = queue.get_next_job()
        if job:
            name, data = job
            print(f"Processing job: {name} (Priority: {next(p for p, q in queue.queues.items() if q and q[0][0] == name)}) - Data: {data}")
        else:
            break

    # Test empty queue
    print("\nQueue empty:", queue.is_empty())
    print("Next job (empty queue):", queue.get_next_job())