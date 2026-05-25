import heapq

   class JobQueue:
       def __init__(self):
           self._heap = []
           self._counter = 0

       def add_job(self, name: str, data: dict, priority: int = 0):
           # Negate priority because heapq is a min-heap
           # Use counter to maintain FIFO order for same priorities
           entry = [-priority, self._counter, name, data]
           self._counter += 1
           heapq.heappush(self._heap, entry)

       def get_next_job(self) -> tuple[str, dict] | None:
           if not self._heap:
               return None
           # Pop the smallest element (highest priority, earliest added)
           priority, counter, name, data = heapq.heappop(self._heap)
           return (name, data)