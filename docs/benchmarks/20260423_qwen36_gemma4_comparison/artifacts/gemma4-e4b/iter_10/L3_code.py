import heapq
from typing import Optional, Tuple, Dict

class JobQueue:
    """
    Implements a priority-based job queue.
    Higher priority numbers (2=Critical, 1=High, 0=Normal) are processed first.
    Jobs of the same priority maintain FIFO order.
    """
    def __init__(self):
        # The heap stores tuples: (-priority, counter, job_name, job_data)
        # Using negative priority converts the min-heap into a max-heap based on priority.
        self._queue = []
        # Counter ensures FIFO order for jobs with the same priority
        self._counter = 0

    def add_job(self, job_name: str, job_data: Dict, priority: int = 0) -> None:
        """
        Adds a job to the queue with a specified priority.
        Priority levels: 0 (Normal), 1 (High), 2 (Critical).
        """
        # We use negative priority because heapq is a min-heap, 
        # and we want the highest positive priority to be considered 'smallest' in the heap.
        heap_element = (-priority, self._counter, job_name, job_data)
        heapq.heappush(self._queue, heap_element)
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """
        Retrieves and removes the highest priority job from the queue.
        Returns a tuple (job_name, job_data) or None if the queue is empty.
        """
        if not self._queue:
            return None
        
        # Pop the smallest element (which corresponds to the highest priority)
        _, _, job_name, job_data = heapq.heappop(self._queue)
        return job_name, job_data

    def is_empty(self) -> bool:
        return not self._queue

    def __len__(self) -> int:
        return len(self._queue)

if __name__ == '__main__':
    # Example Usage Test
    queue = JobQueue()

    # Add jobs with different priorities
    queue.add_job("normal_1", {"type": "normal"}, priority=0)
    queue.add_job("critical_1", {"type": "critical"}, priority=2)
    queue.add_job("high_1", {"type": "high"}, priority=1)
    
    # Add more jobs to test FIFO within priority levels
    queue.add_job("normal_2", {"type": "normal"}, priority=0)
    queue.add_job("critical_2", {"type": "critical"}, priority=2)
    queue.add_job("high_2", {"type": "high"}, priority=1)

    print("--- Processing Jobs ---")
    
    # Expected order: Critical (1st), Critical (2nd), High (1st), High (2nd), Normal (1st), Normal (2nd)

    # 1. Critical (Priority 2)
    job = queue.get_next_job()
    print(f"Processed: {job[0]} (Priority: 2)")

    # 2. Critical (Priority 2) - FIFO check
    job = queue.get_next_job()
    print(f"Processed: {job[0]} (Priority: 2)")

    # 3. High (Priority 1)
    job = queue.get_next_job()
    print(f"Processed: {job[0]} (Priority: 1)")
    
    # 4. High (Priority 1) - FIFO check
    job = queue.get_next_job()
    print(f"Processed: {job[0]} (Priority: 1)")
    
    # 5. Normal (Priority 0)
    job = queue.get_next_job()
    print(f"Processed: {job[0]} (Priority: 0)")

    # 6. Normal (Priority 0) - FIFO check
    job = queue.get_next_job()
    print(f"Processed: {job[0]} (Priority: 0)")

    # Check empty state
    job = queue.get_next_job()
    print(f"Processed: {job}") # Should be None