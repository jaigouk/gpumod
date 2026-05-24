from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   import uuid
   from datetime import datetime

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       func: Callable = field(default=None)
       args: tuple = field(default_factory=tuple)
       kwargs: dict = field(default_factory=dict)
       priority: int = 0
       status: str = "pending"
       created_at: datetime = field(default_factory=datetime.now)
       result: Any = None
       error: Optional[Exception] = None

   class JobQueue:
       def __init__(self):
           self._queue = [] # Will be replaced by PriorityQueue in actual implementation
           # Actually, I should import PriorityQueue here or compose it.
           pass