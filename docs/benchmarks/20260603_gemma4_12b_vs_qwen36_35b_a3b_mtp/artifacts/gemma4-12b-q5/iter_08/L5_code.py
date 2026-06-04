from dataclasses import dataclass, field
    from typing import Callable, Any

    @dataclass(order=True) # order=True helps with PriorityQueue
    class Job:
        priority: int
        id: str
        task: Callable = field(compare=False)
        retries: int = field(default=0, compare=False)