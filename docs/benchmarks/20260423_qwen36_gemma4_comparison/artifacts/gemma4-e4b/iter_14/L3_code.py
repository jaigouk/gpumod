from collections import deque
from typing import Optional, Tuple, Dict, Any

class JobQueue:
    """
    Implements a priority-based job queue.
    Higher priority numbers (2 > 1 > 0) are processed first.
    Jobs of the same priority maintain FIFO order.
    """
    def __init__(self):
        # We use a dictionary where keys are priority levels (0, 1, 2)
        # and values are deques (FIFO queues for that priority).
        self.queues: Dict[int, deque] = {
            0: deque(),  # Normal
            1: deque(),  # High
            2: deque()   # Critical
        }
        self.priority_levels = sorted(self.queues.keys(), reverse=True)

    def add_job(self, job_name: str, job_data: Dict[str, Any], priority: int = 0) -> None:
        """
        Adds a job to the queue with a specified priority.
        """
        if priority not in self.queues:
            raise ValueError(f"Invalid priority level: {priority}. Must be 0, 1, or 2.")
        
        job = {"name": job_name, "data": job_data}
        self.queues[priority].append(job)

    def get_next_job(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Retrieves and removes the highest priority job.
        If multiple jobs share the highest priority, FIFO order is maintained.
        Returns (job_name, job_data) or None if the queue is empty.
        """
        for priority in self.priority_levels:
            if self.queues[priority]:
                job = self.queues[priority].popleft()
                return (job["name"], job["data"])
        
        return None

# Example Usage:
if __name__ == '__main__':
    queue = JobQueue()
    
    print("--- Adding jobs ---")
    # Normal (0) - Should be last
    queue.add_job("job_normal_1", {"type": "normal"}, priority=0)
    
    # Critical (2) - Should be first
    queue.add_job("job_critical_1", {"type": "critical"}, priority=2)
    
    # High (1) - Should be second
    queue.add_job("job_high_1", {"type": "high"}, priority=1)
    
    # Normal (0) - Should be processed after job_normal_1
    queue.add_job("job_normal_2", {"type": "normal"}, priority=0)
    
    # Critical (2) - Should be processed after job_critical_1
    queue.add_job("job_critical_2", {"type": "critical"}, priority=2)

    print("\n--- Processing jobs ---")
    
    while True:
        job = queue.get_next_job()
        if job is None:
            print("Queue is empty.")
            break
        
        name, data = job
        print(f"Processing: {name} (Type: {data['type']})")

    # Expected order: 
    # 1. job_critical_1 (P=2)
    # 2. job_critical_2 (P=2)
    # 3. job_high_1 (P=1)
    # 4. job_normal_1 (P=0)
    # 5. job_normal_2 (P=0)