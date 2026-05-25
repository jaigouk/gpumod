from dataclasses import dataclass, field
   from typing import Any, List, Optional
   from datetime import datetime

   @dataclass
   class Job:
       id: str
       payload: Any
       status: str = "pending"
       created_at: datetime = field(default_factory=datetime.utcnow)
       retries: int = 0

   class JobQueue:
       def __init__(self):
           self._jobs: List[Job] = []

       def add(self, job: Job) -> None:
           self._jobs.append(job)

       def get_next(self) -> Optional[Job]:
           if not self._jobs:
               return None
           return self._jobs.pop(0)

       def is_empty(self) -> bool:
           return len(self._jobs) == 0

       def __len__(self):
           return len(self._jobs)