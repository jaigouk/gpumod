import heapq
   from typing import Optional, Tuple, Dict

   class JobQueue:
       def __init__(self):
           self.heap = []
           self.counter = 0

       def add_job(self, job_id: str, job_data: Dict, priority: int = 0):
           # Higher priority number means higher priority (2 > 1 > 0)
           # Negate priority for min-heap behavior (lower value pops first)
           item = (-priority, self.counter, job_id, job_data)
           heapq.heappush(self.heap, item)
           self.counter += 1

       def get_next_job(self) -> Optional[Tuple[str, dict]]:
           if not self.heap:
               return None
           priority, counter, job_id, job_data = heapq.heappop(self.heap)
           return (job_id, job_data)