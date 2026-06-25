from dataclasses import dataclass
from typing import Callable, Tuple, Optional
from collections import deque
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

    def run(self, fn: Callable, data) -> Tuple[bool, int]:
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
        self._jobs = {}
        self._priority_queues = {}
        self._priorities_heap = []
        self._counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        if job_id in self._jobs:
            old_job = self._jobs[job_id]
            old_prio = old_job.priority
            if old_prio in self._priority_queues:
                dq = self._priority_queues[old_prio]
                self._priority_queues[old_prio] = deque([x for x in dq if x[1] != job_id])
                if not self._priority_queues[old_prio]:
                    del self._priority_queues[old_prio]
        
        self._jobs[job_id] = Job(id=job_id, data=data, priority=priority, retries=0)
        
        if priority not in self._priority_queues:
            self._priority_queues[priority] = deque()
            heapq.heappush(self._priorities_heap, (-priority,))
        
        self._priority_queues[priority].append((self._counter, job_id, data))
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        while self._priorities_heap:
            neg_prio, = heapq.heappop(self._priorities_heap)
            prio = -neg_prio
            
            if prio not in self._priority_queues:
                continue
            
            dq = self._priority_queues[prio]
            if not dq:
                del self._priority_queues[prio]
                continue
            
            order, job_id, data = dq.popleft()
            
            if dq:
                heapq.heappush(self._priorities_heap, (-prio,))
            else:
                del self._priority_queues[prio]
                
            return (job_id, data)
        
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        retry_policy = RetryPolicy(max_attempts=4)
        success, attempts_made = retry_policy.run(lambda d: processor(d), job.data)
        return success