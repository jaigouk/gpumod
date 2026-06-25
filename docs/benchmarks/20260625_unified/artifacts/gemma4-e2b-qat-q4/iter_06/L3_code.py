import heapq

class JobQueue:
    def __init__(self):
        # The heap stores tuples: (-priority, sequence_index, job_name, job_data)
        # We use negative priority because heapq is a min-heap, and we want 
        # higher priority numbers (2) to be processed before lower ones (0).
        # Sequence index (count) handles the FIFO requirement for jobs with the same priority.
        self._queue = []
        self._sequence_counter = 0

    def add_job(self, name: str, job_data: dict, priority: int = 0):
        """
        Adds a job to the queue with the specified priority.
        Priority levels: 0 (Normal), 1 (High), 2 (Critical).
        """
        # We use -priority to simulate a max-priority queue using Python's min-heap.
        # sequence_index ensures FIFO order for jobs of the same priority.
        item = (-priority, self._sequence_counter, name, job_data)
        heapq.heappush(self._queue, item)
        self._sequence_counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves and removes the highest priority job.
        Returns (job_name, job_data) or None if the queue is empty.
        """
        if not self._queue:
            return None

        # Pop the smallest item from the heap
        _, _, name, job_data = heapq.heappop(self._queue)

        return (name, job_data)

if __name__ == "__main__":
    # Example Usage:
    queue = JobQueue()

    # Normal (0)
    queue.add_job("normal", {"type": "normal"}, priority=0)

    # Critical (2)
    queue.add_job("critical", {"type": "critical"}, priority=2)

    # High (1)
    queue.add_job("high", {"type": "high"}, priority=1)

    # Normal (0, arriving later)
    queue.add_job("normal_late", {"type": "normal"}, priority=0)

    print("--- Processing Jobs ---")

    # 1. Critical (Priority 2) should be first
    job1 = queue.get_next_job()
    print(f"Job 1 Retrieved: Name={job1[0]}, Data={job1[1]}") 
    # Output: Name=critical, Data={'type': 'critical'}

    # 2. High (Priority 1) should be second
    job2 = queue.get_next_job()
    print(f"Job 2 Retrieved: Name={job2[0]}, Data={job2[1]}") 
    # Output: Name=high, Data={'type': 'high'}

    # 3. Normal (0, arriving first) should be third
    job3 = queue.get_next_job()
    print(f"Job 3 Retrieved: Name={job3[0]}, Data={job3[1]}")
    # Output: Name=normal, Data={'type': 'normal'}

    # 4. Normal (0, arriving second) should be fourth (FIFO maintained)
    job4 = queue.get_next_job()
    print(f"Job 4 Retrieved: Name={job4[0]}, Data={job4[1]}")
    # Output: Name=normal_late, Data={'type': 'normal'}

    print(f"Next Job (Empty): {queue.get_next_job()}")
    # Output: None