import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Use -priority because heapq is a min-heap, but we want higher priority first.
        # Use self._counter as a tie-breaker to maintain FIFO order for same priorities.
        heapq.heappush(self._heap, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None

        # Pop the entry with the smallest value (highest priority and oldest counter)
        _, _, name, data = heapq.heappop(self._heap)
        return name, data