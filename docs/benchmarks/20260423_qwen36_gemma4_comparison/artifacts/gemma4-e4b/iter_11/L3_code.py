import heapq
from typing import Optional, Tuple, Dict

class JobQueue:
    """
    Implements a priority-based job queue.
    Higher priority number (2 > 1 > 0) means higher urgency.
    Jobs with the same priority maintain FIFO order.
    """
    def __init__(self):
        # The heap stores tuples: (-priority, sequence_number, job_name, job_data)
        # Negating priority allows the min-heap to function as a max-priority queue.
        self.queue: list[Tuple[int, int, str, dict]] = []
        self.counter = 0  # Used for FIFO tie-breaking

    def add_job(self, job_name: str, job_data: dict, priority: int = 0):
        """
        Adds a job to the queue with a specified priority.
        """
        # We negate the priority so that the highest priority number (e.g., 2) 
        # becomes the smallest negative number (-2), ensuring it pops first 
        # from the min-heap.
        heapq.heappush(self.queue, (-priority, self.counter, job_name, job_data))
        self.counter += 1

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        """
        Retrieves and removes the highest priority job from the queue.
        Returns a tuple of (job_name, job_data) or None if the queue is empty.
        """
        if not self.queue:
            return None
        
        # Pop the highest priority item (which is the smallest element in the heap)
        _, _, job_name, job_data = heapq.heappop(self.queue)
        return (job_name, job_data)

# Example Usage:
if __name__ == '__main__':
    queue = JobQueue()

    # 1. Normal job (FIFO test)
    queue.add_job("normal_a", {"type": "normal"}, priority=0)
    
    # 2. Critical job (Highest priority)
    queue.add_job("critical_a", {"type": "critical"}, priority=2)
    
    # 3. High job
    queue.add_job("high_a", {"type": "high"}, priority=1)
    
    # 4. Another normal job (Should come after normal_a)
    queue.add_job("normal_b", {"type": "normal"}, priority=0)

    # 5. Another critical job (FIFO test for critical)
    queue.add_job("critical_b", {"type": "critical"}, priority=2)

    print("--- Processing Jobs ---")
    
    # Expected order: critical_a (P2, first in), critical_b (P2, second in), high_a (P1), normal_a (P0, first in), normal_b (P0, second in)
    
    while True:
        job = queue.get_next_job()
        if job is None:
            break
        name, data = job
        print(f"Processing Job: {name} (Priority: {data.get('priority') if 'priority' in data else 'N/A'})")