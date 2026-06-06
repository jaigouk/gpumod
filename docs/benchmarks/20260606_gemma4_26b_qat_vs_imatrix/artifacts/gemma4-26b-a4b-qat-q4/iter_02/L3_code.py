import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        """
        Adds a job to the queue.
        Higher priority values are processed first.
        Identical priorities follow FIFO order.
        """
        self._counter += 1
        # heapq is a min-heap. We use -priority to simulate a max-heap.
        # We use self._counter to ensure stable sorting (FIFO) for equal priorities.
        heapq.heappush(self._heap, (-priority, self._counter, name, data))

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves the highest priority job from the queue.
        Returns a tuple of (name, data) or None if the queue is empty.
        """
        if not self._heap:
            return None

        _, _, name, data = heapq.heappop(self._heap)
        return name, data