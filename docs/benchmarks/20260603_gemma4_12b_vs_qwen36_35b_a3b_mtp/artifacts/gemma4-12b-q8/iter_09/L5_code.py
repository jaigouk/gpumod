from dataclasses import dataclass
        from collections import deque
        from typing import Any, Optional

        @dataclass
        class Job:
            id: str
            name: str
            payload: Any
            priority: int = 0 # Default for basic queue