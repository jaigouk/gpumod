import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        """
        Adds a job to the queue. 
        Uses -priority to turn heapq (min-heap) into a max-heap for priority.
        Uses self._counter to ensure FIFO for jobs with the same priority.
        """
        heapq.heappush(self._heap, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest priority job as (name, data).
        Returns None if the queue is empty.
        """
        if not self._heap:
            return None

        _, _, name, data = heapq.heappop(self._heap)
        return name, data