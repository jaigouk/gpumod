import heapq
   from collections import deque
   from typing import Optional, Tuple, Dict, Any

   class JobQueue:
       def __init__(self):
           self._heap = []
           self._counter = 0

       def add_job(self, job_id: str, job_data: dict, priority: int = 0) -> None:
           # Use negative priority so higher priority values are popped first
           # Use counter to maintain FIFO order for same priority
           entry = (-priority, self._counter, job_id, job_data)
           heapq.heappush(self._heap, entry)
           self._counter += 1

       def get_next_job(self) -> Optional[Tuple[str, dict]]:
           if not self._heap:
               return None
           _, _, job_id, job_data = heapq.heappop(self._heap)
           return job_id, job_data