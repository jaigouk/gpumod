from dataclasses import dataclass, field
        from typing import Any, List
        from uuid import uuid4

        @dataclass
        class Job:
            payload: Any
            priority: int = 0
            retries: int = 0
            id: str = field(default_factory=lambda: str(uuid4()))