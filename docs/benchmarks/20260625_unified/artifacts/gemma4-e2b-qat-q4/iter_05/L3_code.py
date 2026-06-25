import heapq

class JobQueue:
    """
    Implements a priority-based job scheduler.
    Higher priority values (2 > 1 > 0) are processed first.
    Jobs with the same priority maintain FIFO order.
    """
    def __init__(self):
        # The heap stores tuples: (-priority, insertion_index, job_name, job_data)
        # Negating priority ensures that the highest priority number (e.g., 2) 
        # is treated as the smallest value by the min-heap, causing it to be processed first.
        self._queue = []
        self._insertion_counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        """
        Adds a job to the queue with a specified priority.

        Args:
            name (str): The name of the job.
            data (dict): The data associated with the job.
            priority (int): 0 (Normal), 1 (High), 2 (Critical). Defaults to 0.
        """
        if not (0 <= priority <= 2):
            raise ValueError("Priority must be between 0 and 2.")

        # Push the tuple onto the heap. 
        # We use a negative priority to simulate a max-heap for priority.
        # We use the insertion counter as a tie-breaker for FIFO order.
        entry = (-priority, self._insertion_counter, name, data)
        heapq.heappush(self._queue, entry)
        self._insertion_counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves the highest priority job (Critical first).

        Returns:
            tuple[str, dict] | None: A tuple containing the job name and its data,
                                     or None if the queue is empty.
        """
        if not self._queue:
            return None

        # Pop the smallest element from the heap
        _, _, name, data = heapq.heappop(self._queue)
        return name, data

# Example Usage (for verification)
if __name__ == '__main__':
    queue = JobQueue()
    queue.add_job("normal", {"type": "normal"}, priority=0)
    queue.add_job("critical", {"type": "critical"}, priority=2)
    queue.add_job("high", {"type": "high"}, priority=1)
    queue.add_job("normal_2", {"type": "normal_2"}, priority=0)
    queue.add_job("critical_2", {"type": "critical_2"}, priority=2)

    print(f"--- Retrieving Jobs ---")

    job1 = queue.get_next_job()
    print(f"Job 1 (Highest Priority): {job1}")  # Expected: critical job (priority 2)

    job2 = queue.get_next_job()
    print(f"Job 2: {job2}")  # Expected: high job (priority 1)

    job3 = queue.get_next_job()
    print(f"Job 3 (Normal 1): {job3}")  # Expected: normal job (priority 0, added first)

    job4 = queue.get_next_job()
    print(f"Job 4 (Normal 2): {job4}")  # Expected: normal_2 job (priority 0, added second - FIFO tie-breaker)

    job5 = queue.get_next_job()
    print(f"Job 5 (Critical 2): {job5}")  # Expected: critical_2 job (priority 2)

    job6 = queue.get_next_job()
    print(f"Job 6 (Normal 3): {job6}")  # Expected: normal job (priority 0)

    job7 = queue.get_next_job()
    print(f"Job 7 (Empty): {job7}")  # Expected: None