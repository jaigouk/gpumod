import heapq
from typing import Optional, Tuple, Dict

class JobQueue:
    """
    Implements a priority-based job scheduling queue.
    Higher priority jobs are processed first. Jobs with the same priority
    maintain FIFO order.
    """
    def __init__(self):
        # The heap stores tuples: (-priority, insertion_index, job_name, job_data)
        # Negating priority ensures that higher priority values (e.g., 2)
        # become smaller numbers (-2) and are thus popped first by the min-heap.
        self._queue = []
        self._insertion_counter = 0

    def add_job(self, job_name: str, job_data: Dict, priority: int = 0):
        """
        Adds a job to the queue with a specified priority.

        :param job_name: The identifier of the job.
        :param job_data: Dictionary containing job details.
        :param priority: Priority level (0=Normal, 1=High, 2=Critical).
        """
        # Priority levels: 2 (Critical) > 1 (High) > 0 (Normal)
        # We use negative priority for min-heap behavior (higher priority -> smaller number)
        heapq.heappush(self._queue, (-priority, self._insertion_counter, job_name, job_data))
        self._insertion_counter += 1

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """
        Retrieves and removes the highest priority job from the queue.

        :return: A tuple (job_name, job_data) or None if the queue is empty.
        """
        if not self._queue:
            return None
        
        # Pop the smallest element (which corresponds to highest priority and earliest insertion)
        _, _, job_name, job_data = heapq.heappop(self._queue)
        return job_name, job_data

# Example Usage:
if __name__ == '__main__':
    queue = JobQueue()
    
    # 1. Normal (Priority 0)
    queue.add_job("normal_1", {"type": "normal"}, priority=0)
    
    # 2. Critical (Priority 2)
    queue.add_job("critical_1", {"type": "critical"}, priority=2)
    
    # 3. High (Priority 1)
    queue.add_job("high_1", {"type": "high"}, priority=1)
    
    # 4. Normal (Priority 0) - Should come after normal_1 due to FIFO
    queue.add_job("normal_2", {"type": "normal"}, priority=0)
    
    # 5. Critical (Priority 2) - Should come after critical_1 due to FIFO
    queue.add_job("critical_2", {"type": "critical"}, priority=2)

    print("--- Processing Jobs ---")
    
    # Expected order: critical_1 (2), critical_2 (2), high_1 (1), normal_1 (0), normal_2 (0)
    
    # 1. Critical (2)
    job = queue.get_next_job()
    print(f"Processed: {job[0]} (Priority: 2)")

    # 2. Critical (2) - FIFO tiebreaker
    job = queue.get_next_job()
    print(f"Processed: {job[0]} (Priority: 2)")

    # 3. High (1)
    job = queue.get_next_job()
    print(f"Processed: {job[0]} (Priority: 1)")

    # 4. Normal (0) - FIFO tiebreaker
    job = queue.get_next_job()
    print(f"Processed: {job[0]} (Priority: 0)")
    
    # 5. Normal (0)
    job = queue.get_next_job()
    print(f"Processed: {job[0]} (Priority: 0)")
    
    # Queue is empty
    job = queue.get_next_job()
    print(f"Processed: {job}")