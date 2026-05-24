import heapq
   from collections import deque
   from typing import Optional

   class JobQueue:
       def __init__(self):
           self._queue = []
           self._counter = 0

       def add_job(self, job_id: str, job_data: dict, priority: int = 0):
           # Higher priority first -> negate priority for min-heap
           heapq.heappush(self._queue, (-priority, self._counter, job_id, job_data))
           self._counter += 1

       def get_next_job(self) -> Optional[tuple[str, dict]]:
           if not self._queue:
               return None
           _, _, job_id, job_data = heapq.heappop(self._queue)
           return (job_id, job_data)