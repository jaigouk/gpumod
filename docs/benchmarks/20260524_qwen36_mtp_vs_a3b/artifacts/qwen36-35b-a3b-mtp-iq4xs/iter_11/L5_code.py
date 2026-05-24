from dataclasses import dataclass, field
   from datetime import datetime
   from typing import Any, Optional
   import uuid

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       payload: Any = None
       priority: int = 0
       created_at: datetime = field(default_factory=datetime.utcnow)
       status: str = "pending"  # pending, processing, completed, failed

       def __lt__(self, other):
           # For basic queue comparison if needed
           return self.created_at < other.created_at