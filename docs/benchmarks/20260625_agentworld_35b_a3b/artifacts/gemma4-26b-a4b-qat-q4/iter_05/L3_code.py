import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        """
        Adds a job to the queue.
        Higher priority values are processed first. 
        Same priority jobs follow FIFO.
        """
        # We use -priority because heapq is a min-heap, 
        # and we want higher numbers to be treated as "smaller" to come out first.
        # self._counter ensures FIFO order for jobs with the same priority.
        heapq.heappush(self._heap, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest priority job as (name, data), or None if empty.
        """
        if not self._heap:
            return None

        # Pop the element with the lowest value (highest priority)
        _, _, name, data = heapq.heappop(self._heap)
        return name, data