import heapq

   class JobQueue:
       def __init__(self):
           self._queue = []
           self._counter = 0

       def add_job(self, name, details, priority=0):
           # Use negative priority for max-heap behavior
           # Use counter for FIFO stability
           entry = [-priority, self._counter, name, details]
           self._counter += 1
           heapq.heappush(self._queue, entry)

       def get_next_job(self):
           if not self._queue:
               return None
           # Pop returns the list entry
           neg_priority, count, name, details = heapq.heappop(self._queue)
           return (name, details)