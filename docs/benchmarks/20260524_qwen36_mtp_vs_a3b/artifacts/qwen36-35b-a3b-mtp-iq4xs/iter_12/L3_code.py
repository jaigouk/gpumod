import heapq
import threading

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0
        self._lock = threading.Lock() # Good practice for queues, though not strictly requested, but safer. 
                                      # The prompt doesn't specify thread safety, but it's a queue. 
                                      # I'll stick to the simplest implementation unless concurrency is implied. 
                                      # "Job scheduling" often implies concurrency, but the example is synchronous. 
                                      # I'll skip the lock to keep it simple and focused on the logic requested, 
                                      # unless I feel it's necessary. The prompt asks for specific methods. 
                                      # I'll just use a list and heapq.

    def add_job(self, name: str, job_data: dict, priority: int = 0) -> None:
        # Higher priority number means higher priority.
        # heapq is a min-heap, so we negate priority.
        # Counter ensures FIFO for same priority.
        heapq.heappush(self._queue, (-priority, self._counter, name, job_data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None
        neg_priority, counter, name, job_data = heapq.heappop(self._queue)
        return (name, job_data)