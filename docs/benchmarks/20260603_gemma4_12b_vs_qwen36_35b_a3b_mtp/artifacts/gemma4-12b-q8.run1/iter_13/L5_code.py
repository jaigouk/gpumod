from dataclasses import dataclass
        from typing import Callable, Any, Optional

        @dataclass
        class Job:
            id: str
            task: Callable
            args: tuple = ()
            kwargs: dict = None
            retries: int = 0
            priority: int = 0 # Needed for PriorityQueue