import heapq
   import itertools

   class JobQueue:
       def __init__(self):
           self._queue = []
           self._counter = itertools.count() # or just a simple integer counter

       def add_job(self, job_name: str, job_data: dict, priority: int = 0) -> None:
           # Store as (-priority, counter, job_name, job_data)
           heapq.heappush(self._queue, (-priority, next(self._counter), job_name, job_data))

       def get_next_job(self) -> tuple[str, dict] | None:
           if not self._queue:
               return None
           _, _, job_name, job_data = heapq.heappop(self._queue)
           return (job_name, job_data)