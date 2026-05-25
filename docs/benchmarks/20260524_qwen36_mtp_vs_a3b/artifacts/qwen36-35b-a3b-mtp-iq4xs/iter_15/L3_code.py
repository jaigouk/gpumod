import heapq
   import time
   from typing import Optional, Tuple, Dict

   class JobQueue:
       def __init__(self):
           self._heap = []
           self._counter = 0

       def add_job(self, name: str, data: dict, priority: int = 0) -> None:
           # Use negative priority for max-heap behavior with min-heap
           # Counter ensures FIFO for same priority
           heapq.heappush(self._heap, (-priority, self._counter, name, data))
           self._counter += 1

       def get_next_job(self) -> Optional[Tuple[str, dict]]:
           if not self._heap:
               return None
           _, _, name, data = heapq.heappop(self._heap)
           return (name, data)