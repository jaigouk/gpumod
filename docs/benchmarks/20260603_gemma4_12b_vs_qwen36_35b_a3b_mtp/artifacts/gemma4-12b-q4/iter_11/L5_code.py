from dataclasses import dataclass, field
        from typing import Callable, Any, List
        from collections import deque

        @dataclass
        class Job:
            id: str
            func: Callable
            args: tuple = field(default_factory=tuple)
            kwargs: dict = field(default_factory=dict)
            priority: int = 0
            retries: int = 0
            max_retries: int = 3