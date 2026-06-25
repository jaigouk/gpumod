import heapq

class JobQueue:
    def __init__(self):
        self._heap = []
        self._count = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # We use -priority because heapq is a min-heap, 
        # and we want higher numbers to come out first.
        # self._count ensures FIFO order for jobs with the same priority.
        heapq.heappush(self._heap, (-priority, self._count, name, data))
        self._count += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None

        # Pop the item with the lowest value (which is the highest priority due to negation)
        _, _, name, data = heapq.heappop(self._heap)
        return name, data