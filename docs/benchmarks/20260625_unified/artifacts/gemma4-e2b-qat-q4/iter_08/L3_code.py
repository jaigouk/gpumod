class JobQueue:
    def __init__(self):
        # The heap will store tuples: (-priority, insertion_index, job_name, job_data)
        # Using negative priority ensures the highest priority (2) comes out first in a min-heap.
        self._heap = []
        self._counter = 0

    def add_job(self, job_name: str, job_data: dict, priority: int = 0):
        """
        Adds a job to the queue with a specified priority.
        Higher integer values mean higher priority.
        """
        import heapq
        
        # We store the negative priority to simulate a max-heap behavior for priority
        # and use a counter for stable FIFO ordering.
        heapq.heappush(self._heap, (-priority, self._counter, job_name, job_data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves the highest priority job based on the queue rules.
        Returns (job_name, job_data) or None if the queue is empty.
        """
        import heapq
        
        if not self._heap:
            return None
        
        # Pop the smallest element (which has the highest priority due to negative sorting)
        _, _, job_name, job_data = heapq.heappop(self._heap)
        
        return (job_name, job_data)

if __name__ == '__main__':
    queue = JobQueue()
    queue.add_job("normal", {"type": "normal"}, priority=0)
    queue.add_job("critical", {"type": "critical"}, priority=2)
    queue.add_job("high", {"type": "high"}, priority=1)
    queue.add_job("normal_2", {"type": "normal_2"}, priority=0) # Testing FIFO for priority 0

    print("--- Processing Jobs ---")
    
    # Expected order: critical (2), high (1), normal (0), normal_2 (0)
    
    job1 = queue.get_next_job()
    print(f"Job 1: {job1}")  # Expected: ('critical', {'type': 'critical'})

    job2 = queue.get_next_job()
    print(f"Job 2: {job2}")  # Expected: ('high', {'type': 'high'})

    job3 = queue.get_next_job()
    print(f"Job 3: {job3}")  # Expected: ('normal', {'type': 'normal'})

    job4 = queue.get_next_job()
    print(f"Job 4: {job4}")  # Expected: ('normal_2', {'type': 'normal_2'})

    job5 = queue.get_next_job()
    print(f"Job 5: {job5}")  # Expected: None