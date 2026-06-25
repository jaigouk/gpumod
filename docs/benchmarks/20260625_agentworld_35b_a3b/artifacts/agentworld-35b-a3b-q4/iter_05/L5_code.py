from dataclasses import dataclass
from typing import Callable
import heapq


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
        for attempt in range(self.max_attempts):
            attempts_made = attempt + 1
            try:
                fn(data)
                return (True, attempts_made)
            except Exception:
                pass
        return (False, attempts_made)


class JobQueue:
    def __init__(self):
        self.job_map: dict[str, Job] = {}
        self.priority_queue: list[tuple[int, int, str]] = []
        self.insertion_counter = 0
        self.retry_policy = RetryPolicy(max_attempts=4)

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority, retries=0)
        self.job_map[job_id] = job
        heapq.heappush(self.priority_queue, (-priority, self.insertion_counter, job_id))
        self.insertion_counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        while self.priority_queue:
            neg_priority, order, job_id = heapq.heappop(self.priority_queue)
            if job_id in self.job_map:
                job = self.job_map[job_id]
                return (job.id, job.data)
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.job_map:
            return False
        job = self.job_map[job_id]
        success, attempts = self.retry_policy.run(lambda data: processor(data), job.data)
        if success:
            if job_id in self.job_map:
                del self.job_map[job_id]
        return success