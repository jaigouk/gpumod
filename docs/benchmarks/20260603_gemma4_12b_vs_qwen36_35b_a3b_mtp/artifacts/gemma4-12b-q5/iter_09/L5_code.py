from dataclasses import dataclass, field
        from typing import Any, Optional
        from collections import deque

        @dataclass
        class Job:
            id: str
            task: callable
            priority: int = 0
            retries: int = 0
            max_retries: int = 3
            # ...