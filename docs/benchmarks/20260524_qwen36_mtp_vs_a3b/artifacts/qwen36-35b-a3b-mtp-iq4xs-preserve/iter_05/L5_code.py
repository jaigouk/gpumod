from dataclasses import dataclass, field
     from typing import Any, Callable, Optional
     import queue

     @dataclass
     class Job:
         job_id: str
         func: Callable
         args: tuple = ()
         kwargs: dict = field(default_factory=dict)
         status: str = "pending"  # pending, running, completed, failed

     class JobQueue:
         def __init__(self):
             self._queue = queue.Queue()

         def add(self, job: Job) -> None:
             self._queue.put(job)

         def get(self) -> Optional[Job]:
             try:
                 return self._queue.get_nowait()
             except queue.Empty:
                 return None

         def is_empty(self) -> bool:
             return self._queue.empty()

         def size(self) -> int:
             return self._queue.qsize()