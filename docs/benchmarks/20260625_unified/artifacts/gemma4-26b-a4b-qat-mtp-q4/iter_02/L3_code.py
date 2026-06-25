import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Use -priority because heapq is a min-heap and we want higher numbers first.
        # Use self._counter to ensure FIFO order for jobs with the same priority.
        heapq.heappush(self._heap, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None

        # Pop the job with the highest priority (smallest negative priority)
        _, _, name, data = heapq.heappop(self._heap)
        return name, data