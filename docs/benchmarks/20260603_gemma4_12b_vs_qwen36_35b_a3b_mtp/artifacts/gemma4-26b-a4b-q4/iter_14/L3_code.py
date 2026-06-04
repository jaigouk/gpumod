import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # We negate priority because heapq is a min-heap, but we want a max-priority behavior.
        # We include self._counter to ensure FIFO order for jobs with the same priority.
        heapq.heappush(self._heap, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        
        priority_neg, _, name, data = heapq.heappop(self._heap)
        return (name, data)