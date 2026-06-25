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
        self.jobs_by_id = {}
        self.priority_queues = {}
        self.present_priorities = []
        self.retry_policy = RetryPolicy()

    def _ensure_priority(self, priority: int):
        if priority not in self.priority_queues:
            self.priority_queues[priority] = deque()
            if priority not in self.present_priorities:
                self.present_priorities.append(priority)
                self.present_priorities.sort(key=lambda p: -p)

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        old_priority = None
        if job_id in self.jobs_by_id:
            job = self.jobs_by_id[job_id]
            old_priority = job.priority
            job.data = data
            job.priority = priority
        else:
            self.jobs_by_id[job_id] = Job(id=job_id, data=data, priority=priority, retries=0)
        
        self._ensure_priority(priority)
        dq = self.priority_queues[priority]
        if job_id not in dq:
            dq.append(job_id)
            
        if old_priority is not None and old_priority != priority:
            old_dq = self.priority_queues.get(old_priority)
            if old_dq:
                try:
                    old_dq.remove(job_id)
                except ValueError:
                    pass

    def get_next_job(self) -> tuple[str, dict] | None:
        for priority in self.present_priorities:
            dq = self.priority_queues[priority]
            while dq:
                job_id = dq.popleft()
                if job_id in self.jobs_by_id:
                    job = self.jobs_by_id[job_id]
                    return (job.id, job.data)
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self.jobs_by_id.get(job_id)
        if not job:
            return False
        
        success, attempts = self.retry_policy.run(processor, job.data)
        
        if success:
            if job_id in self.jobs_by_id:
                del self.jobs_by_id[job_id]
            return True
        else:
            job.retries += 1
            return False