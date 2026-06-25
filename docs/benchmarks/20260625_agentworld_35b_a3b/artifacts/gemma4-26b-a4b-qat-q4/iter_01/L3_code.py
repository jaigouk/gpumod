import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # We use -priority because heapq is a min-heap, and we want higher values first.
        # We use self._counter as a tie-breaker to maintain FIFO for equal priorities.
        heapq.heappush(self._heap, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None

        _, _, name, data = heapq.heappop(self._heap)
        return name, data