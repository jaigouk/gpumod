from dataclasses import dataclass
from typing import Callable
from collections import deque

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
        self.jobs: dict[str, Job] = {}
        self.priority_queues: dict[int, deque] = {}
        self.sorted_priorities: list[int] = []
        self.retry_policy = RetryPolicy(max_attempts=4)

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority, retries=0)
        self.jobs[job_id] = job
        
        if priority not in self.priority_queues:
            self.priority_queues[priority] = deque()
            inserted = False
            for i, p in enumerate(self.sorted_priorities):
                if priority > p:
                    self.sorted_priorities.insert(i, priority)
                    inserted = True
                    break
            if not inserted:
                self.sorted_priorities.append(priority)
        
        self.priority_queues[priority].append(job_id)

    def get_next_job(self) -> tuple[str, dict] | None:
        for priority in self.sorted_priorities:
            dq = self.priority_queues.get(priority)
            if dq and len(dq) > 0:
                job_id = dq[0]
                job = self.jobs.get(job_id)
                if job:
                    dq.popleft()
                    if not dq:
                        self.sorted_priorities.remove(priority)
                    return (job.id, job.data)
        
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        success, attempts = self.retry_policy.run(processor, job.data)
        if not success:
            job.retries += 1
        return success