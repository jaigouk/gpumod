from dataclasses import dataclass
from typing import Callable, Dict
from collections import defaultdict, deque


@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0


class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data) -> tuple[bool, int]:
        attempts_made = 0
        for attempt in range(1, self.max_attempts + 1):
            attempts_made = attempt
            try:
                fn(data)
                return (True, attempts_made)
            except Exception:
                pass
        return (False, attempts_made)


class JobQueue:
    def __init__(self):
        self._queues: Dict[int, deque] = defaultdict(deque)
        self._priorities: list[int] = []
        self._job_data: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._job_data[job_id] = data
        if priority not in self._queues:
            inserted = False
            for i, p in enumerate(self._priorities):
                if priority > p:
                    self._priorities.insert(i, priority)
                    inserted = True
                    break
            if not inserted:
                self._priorities.append(priority)
            self._queues[priority] = deque()
        self._queues[priority].append((job_id, data))

    def get_next_job(self) -> tuple[str, dict] | None:
        while self._priorities:
            p = self._priorities[0]
            q = self._queues[p]
            if q:
                job_id, data = q.popleft()
                if not q:
                    self._priorities.pop(0)
                return (job_id, data)
            else:
                self._priorities.pop(0)
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self._job_data.get(job_id)
        if data is None:
            return False
        policy = RetryPolicy()
        success, _ = policy.run(lambda d: processor(d), data)
        return success