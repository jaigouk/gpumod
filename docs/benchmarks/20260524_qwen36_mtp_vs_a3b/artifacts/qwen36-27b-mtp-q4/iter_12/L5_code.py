from dataclasses import dataclass, field
   from typing import Any, Optional
   from collections import deque
   import uuid
   import time

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       payload: Any = None
       priority: int = 0
       status: str = "pending"
       retries: int = 0
       max_retries: int = 3
       created_at: float = field(default_factory=time.time)

   class JobQueue:
       def __init__(self):
           self._queue = deque()
           self._jobs_by_id = {}

       def enqueue(self, job: Job) -> None:
           job.status = "pending"
           self._queue.append(job)
           self._jobs_by_id[job.id] = job

       def dequeue(self) -> Optional[Job]:
           if not self._queue:
               return None
           job = self._queue.popleft()
           job.status = "processing"
           return job

       def peek(self) -> Optional[Job]:
           return self._queue[0] if self._queue else None

       def get_job(self, job_id: str) -> Optional[Job]:
           return self._jobs_by_id.get(job_id)

       def mark_complete(self, job_id: str) -> None:
           if job_id in self._jobs_by_id:
               self._jobs_by_id[job_id].status = "completed"

       def mark_failed(self, job_id: str) -> None:
           if job_id in self._jobs_by_id:
               self._jobs_by_id[job_id].status = "failed"

       def __len__(self) -> int:
           return len(self._queue)