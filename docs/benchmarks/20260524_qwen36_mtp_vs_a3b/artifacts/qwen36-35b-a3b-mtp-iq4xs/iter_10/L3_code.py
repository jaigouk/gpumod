import heapq
   from typing import Optional, Tuple, Dict

   class JobQueue:
       def __init__(self):
           self._heap = []
           self._counter = 0

       def add_job(self, name: str, job_data: dict, priority: int = 0) -> None:
           # Store as (-priority, counter, name, job_data)
           heapq.heappush(self._heap, (-priority, self._counter, name, job_data))
           self._counter += 1

       def get_next_job(self) -> Optional[Tuple[str, dict]]:
           if not self._heap:
               return None
           # Pop the smallest tuple (which corresponds to highest priority, then earliest added)
           _, _, name, job_data = heapq.heappop(self._heap)
           return (name, job_data)