import heapq

class JobQueue:
    """
    Implements a priority-based job scheduler.
    Higher priority numbers mean higher priority.
    Jobs with the same priority maintain FIFO order.
    """
    def __init__(self):
        # Heap stores tuples: (-priority, arrival_index, name, data)
        # Using negative priority ensures that the min-heap structure 
        # behaves like a max-priority queue for priority levels.
        # arrival_index ensures FIFO tie-breaking.
        self.job_heap = []
        self.arrival_counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        """
        Adds a job to the queue with an optional priority.
        priority levels: 0 (Normal), 1 (High), 2 (Critical).
        """
        # Store as (-priority, counter, name, data)
        # The counter is used as a tie-breaker (FIFO).
        heapq.heappush(self.job_heap, (-priority, self.arrival_counter, name, data))
        self.arrival_counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest priority job.
        Returns (job_name, job_data) or None if the queue is empty.
        """
        if not self.job_heap:
            return None
        
        # Pop the smallest element (which corresponds to the highest priority)
        neg_priority, _, name, data = heapq.heappop(self.job_heap)
        
        return name, data

if __name__ == "__main__":
    queue = JobQueue()
    queue.add_job("normal", {"type": "normal"}, priority=0)
    queue.add_job("critical", {"type": "critical"}, priority=2)
    queue.add_job("high", {"type": "high"}, priority=1)
    queue.add_job("normal_2", {"type": "normal_2"}, priority=0) # Should be after normal_1 due to FIFO
    queue.add_job("critical_2", {"type": "critical_2"}, priority=2)
    queue.add_job("low", {"type": "low"}, priority=0)

    print("--- Scheduling Jobs ---")

    # Expect critical (2) first, then critical_2 (2)
    job1 = queue.get_next_job() 
    print(f"Job 1: {job1}") # Expected: critical

    job2 = queue.get_next_job()
    print(f"Job 2: {job2}") # Expected: critical_2
    
    # Expect high (1) next
    job3 = queue.get_next_job()
    print(f"Job 3: {job3}") # Expected: high

    # Expect normal (0) first due to FIFO (arrival_counter)
    job4 = queue.get_next_job()
    print(f"Job 4: {job4}") # Expected: normal

    # Expect normal_2 (0) second due to FIFO
    job5 = queue.get_next_job()
    print(f"Job 5: {job5}") # Expected: normal_2

    # Expect low (0) last
    job6 = queue.get_next_job()
    print(f"Job 6: {job6}") # Expected: low

    # Check empty
    job_empty = queue.get_next_job()
    print(f"Job 7: {job_empty}") # Expected: None