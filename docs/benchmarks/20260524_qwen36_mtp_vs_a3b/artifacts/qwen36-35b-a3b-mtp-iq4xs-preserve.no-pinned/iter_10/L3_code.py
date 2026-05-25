import heapq

   class JobQueue:
       def __init__(self):
           self.heap = []
           self.counter = 0

       def add_job(self, name: str, metadata: dict, priority: int = 0):
           # Higher priority number means higher priority.
           # heapq is a min-heap, so we negate priority to get max-heap behavior.
           # Tuple: (-priority, counter, name, metadata)
           # counter ensures FIFO for same priority.
           heapq.heappush(self.heap, (-priority, self.counter, name, metadata))
           self.counter += 1

       def get_next_job(self) -> tuple[str, dict] | None:
           if not self.heap:
               return None
           
           # Pop the smallest item (which is the highest priority due to negation)
           # Tuple structure: (-priority, counter, name, metadata)
           _, _, name, metadata = heapq.heappop(self.heap)
           return (name, metadata)