from dataclasses import dataclass
from typing import Callable

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: dict) -> tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                fn(data)
                return True, attempts + 1
            except Exception:
                attempts += 1
        return False, attempts

class JobQueue:
    def __init__(self):
        self.jobs = []
        self.retry_policy = RetryPolicy()
        self.counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self.jobs.append({
            'job': Job(job_id, data, priority),
            'seq': self.counter
        })
        self.counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        for item in self.jobs:
            if item['job'].id == job_id:
                success, _ = self.retry_policy.run(processor, item['job'].data)
                return success
        return False

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None

        # Sort by highest priority first (desc), then by sequence number (asc) for FIFO
        self.jobs.sort(key=lambda x: (-x['job'].priority, x['seq']))
        entry = self.jobs.pop(0)
        return (entry['job'].id, entry['job'].data))