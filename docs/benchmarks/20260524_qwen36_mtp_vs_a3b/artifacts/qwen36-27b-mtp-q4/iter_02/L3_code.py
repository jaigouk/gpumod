import heapq
   from typing import Optional

   class JobQueue:
       def __init__(self):
           self._queue = []
           self._counter = 0

       def add_job(self, name: str, data: dict, priority: int = 0) -> None:
           # Use negative priority so higher priority values come first in min-heap
           # Use counter to maintain FIFO for same priority
           heapq.heappush(self._queue, (-priority, self._counter, name, data))
           self._counter += 1

       def get_next_job(self) -> Optional[tuple[str, dict]]:
           if not self._queue:
               return None
           _, _, name, data = heapq.heappop(self._queue)
           return (name, data)