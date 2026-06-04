from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass
        class Job:
            id: str
            payload: Any
            priority: int = 0
            retries: int = 0