import heapq

class JobQueue:
    """
    Implements a priority-based job scheduling queue.
    Priority levels: 2 (Critical) > 1 (High) > 0 (Normal).
    Jobs with the same priority maintain FIFO order.
    """
    def __init__(self):
        # Min-heap storing elements: (-priority, counter, (name, data))
        # Using negative priority ensures that higher priority values (e.g., 2) 
        # result in a smaller numerical value, thus popping first.
        # The counter is used for FIFO tie-breaking.
        self.heap = []
        self.counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        """
        Adds a job to the queue with an optional priority.
        :param name: Identifier of the job.
        :param data: Payload dictionary for the job.
        :param priority: Priority level (0=Normal, 1=High, 2=Critical).
        """
        # Store negative priority because heapq is a min-heap, 
        # and we want high numbers (2) processed before low numbers (0).
        entry = (-priority, self.counter, (name, data))
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves the highest priority job from the queue.
        Returns None if the queue is empty.
        """
        if not self.heap:
            return None

        # Pop the entry with the highest priority
        _, _, (name, data) = heapq.heappop(self.heap)
        return (name, data)

if __name__ == '__main__':
    queue = JobQueue()

    # Adding jobs
    queue.add_job("normal", {"type": "normal"}, priority=0)
    queue.add_job("critical", {"type": "critical"}, priority=2)
    queue.add_job("high", {"type": "high"}, priority=1)
    queue.add_job("normal_2", {"type": "normal"}, priority=0)
    queue.add_job("critical_2", {"type": "critical"}, priority=2)

    # Testing retrieval order: Critical (2), Critical (2), High (1), Normal (0), Normal (0)
    print(f"Processing Job 1: {queue.get_next_job()}")  # Should be Critical (P2)
    print(f"Processing Job 2: {queue.get_next_job()}")  # Should be Critical (P2) - FIFO tie-break
    print(f"Processing Job 3: {queue.get_next_job()}")  # Should be High (P1)
    print(f"Processing Job 4: {queue.get_next_job()}")  # Should be Normal (P0) - FIFO tie-break