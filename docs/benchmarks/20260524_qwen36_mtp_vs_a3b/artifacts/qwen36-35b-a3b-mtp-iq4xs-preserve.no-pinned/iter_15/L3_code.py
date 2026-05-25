import heapq

   class JobQueue:
       def __init__(self):
           self.queue = []
           self.counter = 0  # For FIFO tie-breaking

       def add_job(self, job_id: str, job_data: dict, priority: int = 0):
           # heapq is a min-heap. To get highest priority first, we negate priority.
           # Tuple comparison compares first element, then second.
           # (-priority, counter) ensures higher priority comes first, then lower counter (FIFO).
           entry = (-priority, self.counter, job_id, job_data)
           self.counter += 1
           heapq.heappush(self.queue, entry)

       def get_next_job(self) -> tuple[str, dict] | None:
           if not self.queue:
               return None
           
           # Pop the smallest item (which is highest priority due to negation)
           _, _, job_id, job_data = heapq.heappop(self.queue)
           return (job_id, job_data)