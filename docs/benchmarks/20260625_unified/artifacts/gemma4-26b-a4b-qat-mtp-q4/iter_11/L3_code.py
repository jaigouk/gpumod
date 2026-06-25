import heapq
import itertools

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # We use -priority because heapq is a min-heap, and we want higher priority first.
        # We use next(self._counter) to ensure FIFO order for jobs with the same priority.
        heapq.heappush(self._heap, (-priority, next(self._counter), name, data))

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None

        _, _, name, data = heapq.heappop(self._heap)
        return name, data