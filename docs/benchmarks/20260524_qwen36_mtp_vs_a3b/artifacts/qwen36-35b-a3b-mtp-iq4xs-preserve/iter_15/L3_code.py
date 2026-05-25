import heapq
   import itertools

   class JobQueue:
       def __init__(self):
           self._queue = []
           self._counter = itertools.count()  # Monotonically increasing counter

       def add_job(self, name: str, data: dict, priority: int = 0):
           # Negate priority so higher priority comes first in min-heap
           entry = (-priority, next(self._counter), name, data)
           heapq.heappush(self._queue, entry)

       def get_next_job(self) -> tuple[str, dict] | None:
           if not self._queue:
               return None
           _, _, name, data = heapq.heappop(self._queue)
           return (name, data)