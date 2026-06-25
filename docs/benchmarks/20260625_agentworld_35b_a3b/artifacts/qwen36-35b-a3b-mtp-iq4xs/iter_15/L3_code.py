import heapq

class JobQueue:
    def __init__(self) -> None:
        self._queue: list = []
        self._counter: int = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None
        _, _, name, data = heapq.heappop(self._queue)
        return name, data