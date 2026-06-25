import heapq

class JobQueue:
    """
    Implements a priority-based job scheduling queue using a min-heap.
    Higher priority numbers mean higher priority jobs (e.g., 2 > 1 > 0).
    FIFO is maintained for jobs with the same priority using a sequence counter.
    """
    def __init__(self):
        # The heap will store tuples: (-priority, sequence, name, data)
        # We negate the priority because heapq is a min-heap, but we want 
        # higher priority numbers (like 2) to float to the top.
        self.queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        """
        Adds a job to the queue with a specified priority.

        :param name: Identifier for the job.
        :param data: Dictionary containing job details.
        :param priority: Priority level (0=Normal, 1=High, 2=Critical). Defaults to 0.
        """
        # The sequence counter ensures FIFO ordering for jobs with the same priority.
        self.queue.append((-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves the highest priority job from the queue.

        :return: A tuple (name, data) of the job, or None if the queue is empty.
        """
        if not self.queue:
            return None

        # Pop the smallest element from the heap.
        # The smallest element will be the one with the largest negative priority, 
        # meaning the highest actual priority.
        _, _, name, data = heapq.heappop(self.queue)

        return (name, data)

if __name__ == '__main__':
    queue = JobQueue()

    # Adding jobs:
    # Priority 0 (Normal)
    queue.add_job("normal", {"type": "normal"}, priority=0)

    # Priority 2 (Critical)
    queue.add_job("critical", {"type": "critical"}, priority=2)

    # Priority 1 (High)
    queue.add_job("high", {"type": "high"}, priority=1)

    # Another Priority 0 job
    queue.add_job("normal_2", {"type": "normal"}, priority=0)

    print("--- Processing Jobs ---")

    # Expected order: critical (2) -> high (1) -> normal (0, first one) -> normal (0, second one)

    job1 = queue.get_next_job()
    print(f"Next job: {job1}")  # Expected: ('critical', {'type': 'critical'})

    job2 = queue.get_next_job()
    print(f"Next job: {job2}")  # Expected: ('high', {'type': 'high'})

    job3 = queue.get_next_job()
    print(f"Next job: {job3}")  # Expected: ('normal', {'type': 'normal'})

    job4 = queue.get_next_job()
    print(f"Next job: {job4}")  # Expected: ('normal', {'type': 'normal'})

    job5 = queue.get_next_job()
    print(f"Next job: {job5}")  # Expected: None