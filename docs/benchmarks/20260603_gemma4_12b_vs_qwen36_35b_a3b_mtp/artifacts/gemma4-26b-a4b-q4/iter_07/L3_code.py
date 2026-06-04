import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        """
        Adds a job to the queue. 
        Uses negative priority for max-heap behavior and a counter for FIFO.
        """
        # heapq is a min-heap. By negating priority, the highest number 
        # becomes the smallest number, ensuring it is processed first.
        # The counter ensures that jobs with the same priority maintain FIFO order.
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves and removes the highest priority job from the queue.
        Returns None if the queue is empty.
        """
        if not self._queue:
            return None

        _, _, name, data = heapq.heappop(self._queue)
        return name, data