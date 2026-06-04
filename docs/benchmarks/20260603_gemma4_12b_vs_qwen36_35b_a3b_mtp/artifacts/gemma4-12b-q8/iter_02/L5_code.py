from dataclasses import dataclass, field
        from typing import Any
        from enum import IntEnum

        class Priority(IntEnum):
            LOW = 3
            MEDIUM = 2
            HIGH = 1

        @dataclass(order=True) # Order=True is useful for PriorityQueue
        class Job:
            priority: Priority
            payload: Any
            id: str = field(default="", metadata={"compare": False})
            retries: int = 0
            max_retries: int = 3