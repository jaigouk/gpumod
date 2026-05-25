"""Core job data structure and basic queue operations."""
   from dataclasses import dataclass, field
   from typing import Any, Callable, List, Optional
   from collections import deque

   @dataclass
   class Job:
       """Represents a unit of work."""
       func: Callable
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       job_id: Optional[str] = None
       status: str = "pending"  # pending, running, completed, failed
       priority: int = 0
       result: Any = None
       error: Optional[Exception] = None

   class JobQueue:
       """Basic FIFO job queue."""
       def __init__(self):
           self._queue: deque[Job] = deque()

       def enqueue(self, job: Job) -> None:
           """Add a job to the end of the queue."""
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           """Remove and return the next job."""
           return self._queue.popleft() if self._queue else None

       def peek(self) -> Optional[Job]:
           """View the next job without removing it."""
           return self._queue[0] if self._queue else None

       def is_empty(self) -> bool:
           """Check if the queue is empty."""
           return len(self._queue) == 0

       def size(self) -> int:
           """Return the number of jobs in the queue."""
           return len(self._queue)

       def clear(self) -> None:
           """Remove all jobs from the queue."""
           self._queue.clear()