import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, params: dict, priority: int = 0) -> None:
        heapq.heappush(self._queue, (-priority, self._counter, name, params))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None
        _, _, name, params = heapq.heappop(self._queue)
        return name, params