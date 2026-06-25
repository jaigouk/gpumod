import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # heapq is a min-heap. To treat higher numbers as higher priority,
        # we store the negative of the priority.
        # self._counter acts as a tie-breaker to maintain FIFO order for same priority.
        heapq.heappush(self._heap, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None

        # Pop the element with the lowest -priority (highest actual priority)
        _, _, name, data = heapq.heappop(self._heap)
        return name, data