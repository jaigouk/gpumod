from dataclasses import dataclass, field
   from datetime import datetime
   from typing import Any, Callable, List, Optional
   import uuid

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       task: Optional[Callable] = None
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       status: str = "pending"
       created_at: datetime = field(default_factory=datetime.utcnow)
       result: Any = None
       error: Optional[str] = None

   class JobQueue:
       def __init__(self):
           self._jobs: List[Job] = []

       def enqueue(self, job: Job) -> None:
           self._jobs.append(job)

       def dequeue(self) -> Optional[Job]:
           if self.is_empty():
               return None
           return self._jobs.pop(0)

       def peek(self) -> Optional[Job]:
           if self.is_empty():
               return None
           return self._jobs[0]

       def is_empty(self) -> bool:
           return len(self._jobs) == 0

       def size(self) -> int:
           return len(self._jobs)

       def process_all(self, processor: Callable[[Job], Any]) -> List[Job]:
           processed = []
           while not self.is_empty():
               job = self.dequeue()
               try:
                   job.status = "processing"
                   job.result = processor(job)
                   job.status = "completed"
               except Exception as e:
                   job.status = "failed"
                   job.error = str(e)
               processed.append(job)
           return processed