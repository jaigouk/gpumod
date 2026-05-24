import heapq
   from typing import Optional

   class JobQueue:
       def __init__(self):
           self._queue = []
           self._counter = 0

       def add_job(self, job_name: str, job_data: dict, priority: int = 0) -> None:
           # Use negative priority for max-heap behavior
           entry = (-priority, self._counter, job_name, job_data)
           heapq.heappush(self._queue, entry)
           self._counter += 1

       def get_next_job(self) -> Optional[tuple[str, dict]]:
           if self._queue:
               _, _, job_name, job_data = heapq.heappop(self._queue)
               return job_name, job_data
           return None