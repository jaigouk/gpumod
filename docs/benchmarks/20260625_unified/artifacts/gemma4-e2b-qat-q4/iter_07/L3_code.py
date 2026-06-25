import heapq

class JobQueue:
    def __init__(self):
        # The heap will store tuples of: (-priority, sequence_index, job_data)
        self._queue = []
        self._sequence_counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Convert priority to negative because heapq is a min-heap,
        # and we want higher priority numbers (e.g., 2) to come out first.
        neg_priority = -priority

        # Push the tuple onto the heap. 
        # Using neg_priority ensures higher priority is treated as smaller in the heap.
        # Sequence counter ensures FIFO for jobs with the same priority.
        entry = (neg_priority, self._sequence_counter, name, data)
        heapq.heappush(self._queue, entry)
        self._sequence_counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        # heappop retrieves the smallest element. 
        # This corresponds to the highest negative priority (i.e., highest positive priority).
        neg_priority, _, name, data = heapq.heappop(self._queue)

        return name, data

if __name__ == '__main__':
    queue = JobQueue()

    # Test Case 1: Standard insertion
    print(f"Adding Normal (P=0)")
    queue.add_job("normal", {"type": "normal"}, priority=0)

    print(f"Adding Critical (P=2)")
    queue.add_job("critical", {"type": "critical"}, priority=2)

    print(f"Adding High (P=1)")
    queue.add_job("high", {"type": "high"}, priority=1)

    # Result should be Critical (P=2)
    job1 = queue.get_next_job()
    print(f"\nNext Job: {job1}") 

    # Result should be High (P=1)
    job2 = queue.get_next_job()
    print(f"Next Job: {job2}")

    # Result should be Normal (P=0)
    job3 = queue.get_next_job()
    print(f"Next Job: {job3}")

    print("\n--- Testing FIFO tie-breaker ---")
    queue2 = JobQueue()
    # Both have priority 0. 'job_a' added first.
    queue2.add_job("job_a", {"type": "normal"}, priority=0)
    queue2.add_job("job_b", {"type": "normal"}, priority=0)
    # 'job_c' added second.
    queue2.add_job("job_c", {"type": "normal"}, priority=0)

    job_a = queue2.get_next_job()
    print(f"FIFO Test 1: {job_a}") # Expected: job_a

    job_b = queue2.get_next_job()
    print(f"FIFO Test 2: {job_b}") # Expected: job_b