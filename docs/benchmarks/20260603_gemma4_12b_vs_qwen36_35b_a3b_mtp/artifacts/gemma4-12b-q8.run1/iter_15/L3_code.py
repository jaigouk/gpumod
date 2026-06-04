import heapq

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # Use negative priority to turn min-heap into max-heap for priority levels
        # Use self._counter to maintain FIFO order for equal priorities
        heapq.heappush(self._queue, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None
        
        _, _, name, data = heapq.heappop(self._queue)
        return (name, data)