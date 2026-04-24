import heapq
from typing import Optional, Tuple, Dict

class JobQueue:
    """
    Implements a priority-based job scheduling queue.
    Higher priority numbers mean higher urgency (2 > 1 > 0).
    Jobs with the same priority maintain FIFO order.
    """
    def __init__(self):
        # The heap stores tuples: (-priority, insertion_index, job_name, job_data)
        # Using -priority ensures that higher priority jobs (e.g., 2) have a smaller
        # value (-2) and thus are processed first by the min-heap.
        self._queue = []
        self._counter = 0

    def add_job(self, job_name: str, job_data: Dict, priority: int = 0):
        """
        Adds a job to the queue with a specified priority.
        Priority levels: 0 (Normal), 1 (High), 2 (Critical).
        """
        # Use negative priority for min-heap sorting (higher priority = smaller negative value)
        # Use counter for FIFO tie-breaking
        heapq.heappush(self._queue, (-priority, self._counter, job_name, job_data))
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """
        Retrieves and removes the highest priority job from the queue.
        Returns (job_name, job_data) or None if the queue is empty.
        """
        if not self._queue:
            return None
        
        # Pop the smallest item (which corresponds to the highest priority/earliest insertion)
        _, _, job_name, job_data = heapq.heappop(self._queue)
        return (job_name, job_data)

# Example Usage:
if __name__ == '__main__':
    queue = JobQueue()

    # 1. Normal job (P=0) - Should be processed last among the initial set
    queue.add_job("normal_A", {"type": "normal"}, priority=0) 

    # 2. Critical job (P=2) - Should be processed first
    queue.add_job("critical_B", {"type": "critical"}, priority=2) 

    # 3. High job (P=1) - Should be processed second
    queue.add_job("high_C", {"type": "high"}, priority=1)

    # 4. Another normal job (P=0) - Should maintain FIFO order relative to normal_A
    queue.add_job("normal_D", {"type": "normal"}, priority=0)

    # 5. Another critical job (P=2) - Should be processed after critical_B
    queue.add_job("critical_E", {"type": "critical"}, priority=2)


    print("--- Processing Jobs ---")
    while True:
        job = queue.get_next_job()
        if job is None:
            break
        name, data = job
        print(f"Processing: {name} (Type: {data['type']})")

    # Expected order: critical_B, critical_E, high_C, normal_A, normal_D