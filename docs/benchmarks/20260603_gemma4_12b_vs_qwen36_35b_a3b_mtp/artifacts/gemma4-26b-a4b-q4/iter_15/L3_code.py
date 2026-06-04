import heapq

    class JobQueue:
        def __init__(self):
            self._jobs = []
            self._counter = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Use negative priority for max-heap behavior using heapq (min-heap)
            # Use self._counter to ensure FIFO for same priority
            heapq.heappush(self._jobs, (-priority, self._counter, name, data))
            self._counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._jobs:
                return None
            _, _, name, data = heapq.heappop(self._jobs)
            return name, data