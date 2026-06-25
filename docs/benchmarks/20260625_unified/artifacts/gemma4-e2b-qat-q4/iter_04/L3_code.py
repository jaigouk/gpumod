import heapq

class JobQueue:
    """
    Implements a priority-based job scheduler where higher priority numbers 
    are processed first, and FIFO order is maintained for jobs with the same priority.
    """
    def __init__(self):
        # The priority queue stores tuples: (-priority, insertion_index, job_name, job_details)
        # The insertion_index ensures FIFO ordering for jobs with the same priority.
        self.queue = []
        self.counter = 0

    def add_job(self, job_name: str, job_details: dict, priority: int = 0):
        """
        Adds a job to the queue with an optional priority.
        Higher priority numbers are processed first.
        """
        # We negate the priority because heapq is a min-heap, and we want 
        # the highest number (Critical=2) to have the lowest value for the heap.
        # insertion_index acts as a tie-breaker (FIFO).
        item = (-priority, self.counter, job_name, job_details)
        heapq.heappush(self.queue, item)
        self.counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves and removes the highest priority job.

        Returns:
            A tuple containing (job_name, job_details), or None if the queue is empty.
        """
        if not self.queue:
            return None

        # Pop the item with the smallest value (highest priority)
        _, _, job_name, job_details = heapq.heappop(self.queue)

        return (job_name, job_details)

if __name__ == '__main__':
    # Example Usage:
    queue = JobQueue()
    queue.add_job("normal", {"type": "normal"}, priority=0)
    queue.add_job("critical", {"type": "critical"}, priority=2)
    queue.add_job("high", {"type": "high"}, priority=1)
    queue.add_job("normal_2", {"type": "normal_2"}, priority=0)
    queue.add_job("critical_2", {"type": "critical_2"}, priority=2)

    print("Job 1 (Expected Critical):", queue.get_next_job())
    print("Job 2 (Expected Critical):", queue.get_next_job())
    print("Job 3 (Expected High):", queue.get_next_job())
    print("Job 4 (Expected Normal 1):", queue.get_next_job())
    print("Job 5 (Expected Normal 2):", queue.get_next_job())
    print("Job 6 (Expected None):", queue.get_next_job())