import heapq

class JobQueue:
    def __init__(self):
        self._jobs = []
        self._counter = 0

    def add_job(self, name: str, data: dict, priority: int = 0) -> None:
        # We use -priority because heapq is a min-heap, and we want higher 
        # priority numbers to be processed first.
        # self._counter ensures FIFO order for jobs with the same priority.
        heapq.heappush(self._jobs, (-priority, self._counter, name, data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._jobs:
            return None

        _, _, name, data = heapq.heappop(self._jobs)
        return name, data