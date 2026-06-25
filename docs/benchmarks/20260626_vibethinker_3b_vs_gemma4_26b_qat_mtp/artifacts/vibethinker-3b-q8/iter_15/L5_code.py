from dataclass import dataclass
from collections import defaultdict, deque
from typing import Callable, Tuple, Dict, Any, Optional

@dataclass
class Job:
    id: str
    data: Dict[str, Any]
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    __slots__ = ("max_attempts",)

    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable[[], Any], data: Dict[str, Any]) -> Tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                fn(data)
                return True, attempts + 1
            except Exception:
                attempts += 1
                continue
        return False, self.max_attempts

class JobQueue:
    __slots__ = ("_jobs", "_retry_policy")

    def __init__(self):
        self._jobs: Dict[int, deque[Job]] = defaultdict(deque)
        self._retry_policy = RetryPolicy(max_attempts=4)

    def add_job(self, job_id: str, data: Dict[str, Any], priority: int = 0) -> None:
        job = Job(job_id, data, priority, 0)
        self._jobs[priority].append(job)

    def process_job(self, job_id: str, processor: Callable[[], Any]) -> bool:
        # Retrieve the next job (ignoring the supplied id)
        job_id_actual, job_data = self._get_next_job()
        if job_id_actual is None:
            return False
        success, _ = self._retry_policy.run(processor, job_data)
        return success

    def _get_next_job(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        for priority in sorted(self._jobs.keys()):
            if self._jobs[priority]:
                job = self._jobs[priority].popleft()
                return job.id, job.data
        return None