import heapq
   from typing import Optional, Tuple, Dict

   class JobQueue:
       def __init__(self):
           self._queue = []
           self._counter = 0

       def add_job(self, name: str, params: dict, priority: int = 0) -> None:
           # Use negative priority so higher priority comes first in min-heap
           # Use counter to maintain FIFO for same priority
           heapq.heappush(self._queue, (-priority, self._counter, name, params))
           self._counter += 1

       def get_next_job(self) -> Optional[Tuple[str, dict]]:
           if self._queue:
               _, _, name, params = heapq.heappop(self._queue)
               return name, params
           return None