from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass(order=True) # order=True helps with priority if implemented correctly
        class Job:
            priority: int
            id: str = field(compare=False)
            data: Any = field(compare=False)

        class JobQueue:
            def push(self, job: Job): raise NotImplementedError
            def pop(self) -> Job: raise NotImplementedError
            def is_empty(self) -> bool: raise NotImplementedError