import time
   import uuid
   from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   from collections import deque

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       payload: Any = None
       priority: int = 0
       status: str = "pending"
       retry_count: int = 0
       max_retries: int = 3

   class JobQueue:
       def __init__(self):
           self._queue = deque()

       def enqueue(self, job: Job) -> None:
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           if self._queue:
               return self._queue.popleft()
           return None

       def is_empty(self) -> bool:
           return len(self._queue) == 0

   class PriorityQueue(JobQueue):
       def enqueue(self, job: Job) -> None:
           # Insert in sorted order by priority (lower number = higher priority)
           inserted = False
           for i in range(len(self._queue)):
               if job.priority < self._queue[i].priority:
                   self._queue.insert(i, job)
                   inserted = True
                   break
           if not inserted:
               self._queue.append(job)

   def process_with_retry(job: Job, processor: Callable[[Job], Any], base_delay: float = 1.0) -> Any:
       last_exception = None
       while job.retry_count <= job.max_retries:
           try:
               job.status = "running"
               result = processor(job)
               job.status = "completed"
               return result
           except Exception as e:
               job.retry_count += 1
               last_exception = e
               if job.retry_count > job.max_retries:
                   job.status = "failed"
                   raise
               delay = base_delay * (2 ** (job.retry_count - 1))
               time.sleep(delay)
       raise last_exception