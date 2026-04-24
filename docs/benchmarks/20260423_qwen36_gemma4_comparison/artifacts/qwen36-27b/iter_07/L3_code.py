import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        entry = (-priority, self._counter, name, data)
        heapq.heappush(self._heap, entry)
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if self._heap:
            _, _, name, data = heapq.heappop(self._heap)
            return name, data
        return None