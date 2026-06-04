import heapq
    from typing import Any

    class JobQueue:
        def __init__(self):
            self._queue = []
            self._counter = 0

        def add_job(self, name: str, data: dict[str, Any], priority: int = 0) -> None:
            # heapq is a min-heap. To make higher priority numbers come first,
            # we store the negative of the priority.
            # The counter ensures FIFO for same priorities.
            heapq.heappush(self._queue, (-priority, self._counter, name, data))
            self._counter += 1

        def get_next_job(self) -> tuple[str, dict[str, Any]] | None:
            if not self._queue:
                return None
            _, _, name, data = heapq.heappop(self._queue)
            return name, data