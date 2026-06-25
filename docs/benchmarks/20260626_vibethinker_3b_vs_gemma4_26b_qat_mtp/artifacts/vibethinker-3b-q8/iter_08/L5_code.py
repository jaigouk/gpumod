from typing import Callable, Tuple, Dict
from collections import deque
from dataclass import dataclass

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data) -> Tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                fn(data)
                return (True, attempts + 1)
            except Exception:
                attempts += 1
                continue
        return (False, self.max_attempts)

class JobQueue:
    def __init__(self):
        self.priority_q: Dict[int, deque[Job]] = {}
        self.job_by_id: Dict[str, Job] = {}
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(job_id, data, priority, 0)
        self.job_by_id[job_id] = job
        if priority not in self.priority_q:
            self.priority_q[priority] = deque()
        self.priority_q[priority].append(job)

    def get_next_job(self) -> Tuple[str, dict] | None:
        for priority in sorted(self.priority_q.keys()):
            if self.priority_q[priority]:
                job = self.priority_q[priority].popleft()
                self.job_by_id.pop(job.id, None)
                return (job.id, job.data)
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self.job_by_id.pop(job_id)
        priority = job.priority
        queue = self.priority_q.get(priority)
        if queue:
            queue.remove(job)
        success, attempts = self.retry_policy.run(processor, job.data)
        return success