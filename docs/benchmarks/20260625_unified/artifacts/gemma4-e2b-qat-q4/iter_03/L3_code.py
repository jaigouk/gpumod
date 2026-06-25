import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._insertion_counter = 0

    def add_job(self, name: str, details: dict, priority: int = 0):
        """
        Adds a job to the queue.

        Args:
            name: The identifier for the job.
            details: The payload of the job.
            priority: The priority level (0=Normal, 1=High, 2=Critical).
        """
        self._insertion_counter += 1
        # Use -priority for max-heap behavior (highest priority number comes first)
        # Use insertion_counter as the tie-breaker for FIFO (lower counter added first)
        heapq.heappush(self._queue, (-priority, self._insertion_counter, (name, details)))

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves the highest priority job.

        Returns:
            A tuple (job_name, job_details) or None if the queue is empty.
        """
        if not self._queue:
            return None

        _, _, (name, details) = heapq.heappop(self._queue)
        return (name, details)

if __name__ == '__main__':
    # Test case 1: Basic functionality
    queue = JobQueue()
    queue.add_job("normal", {"type": "normal"}, priority=0)
    queue.add_job("critical", {"type": "critical"}, priority=2)
    queue.add_job("high", {"type": "high"}, priority=1)
    queue.add_job("another_normal", {"type": "normal"}, priority=0)

    job = queue.get_next_job()
    print(f"Job 1: {job}")  # Expected: critical (Priority 2)

    job = queue.get_next_job()
    print(f"Job 2: {job}")  # Expected: high (Priority 1)

    job = queue.get_next_job()
    print(f"Job 3: {job}")  # Expected: another_normal (Priority 0, FIFO)

    job = queue.get_next_job()
    print(f"Job 4: {job}")  # Expected: normal (Priority 0)