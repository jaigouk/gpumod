from dataclasses import dataclass
        from typing import Callable, Any

        @dataclass
        class Job:
            id: str
            task: Callable[[], Any]
            priority: int = 0
            retries: int = 0
            max_retries: int = 3