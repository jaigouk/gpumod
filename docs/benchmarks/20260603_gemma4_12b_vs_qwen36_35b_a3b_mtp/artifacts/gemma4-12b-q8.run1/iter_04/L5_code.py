from dataclasses import dataclass
        from typing import Any, List

        @dataclass
        class Job:
            id: str
            payload: Any
            retries: int = 0

        class JobQueue:
            def __init__(self):
                self.queue = []
            def put(self, job: Job): ...
            def get(self): ...