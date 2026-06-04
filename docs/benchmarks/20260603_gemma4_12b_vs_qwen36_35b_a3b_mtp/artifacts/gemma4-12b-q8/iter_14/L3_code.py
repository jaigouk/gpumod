import heapq

    class JobQueue:
        def __init__(self):
            self._heap = []
            self._counter = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # priority is 0, 1, 2. 
            # heapq is min-heap, so we use -priority to make 2 > 1 > 0.
            heapq.heappush(self._heap, (-priority, self._counter, name, data))
            self._counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self._heap:
                return None
            _, _, name, data = heapq.heappop(self._heap)
            return (name, data)