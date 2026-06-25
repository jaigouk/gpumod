from dataclass import dataclass
from typing import Callable, Tuple, List, Optional, Dict, Any

@dataclass
class Job:
    id: str
    data: Dict[str, Any]
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable[[], Any], data: Any) -> Tuple[bool, int]:
        attempt = 0
        while attempt < self.max_attempts:
            try:
                fn(data)
                return True, attempt + 1
            except Exception:
                attempt += 1
                continue
        return False, attempt

class JobQueue:
    def __init__(self):
        self._jobs: List[Tuple[Any, Job]] = []
        self._timestamp = 0

    def add_job(self, job_id: str, data: Dict[str, Any], priority: int = 0) -> None:
        job = Job(job_id, data, priority, 0)
        self._jobs.append((self._timestamp, job))
        self._timestamp += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        for entry in self._jobs:
            job = entry[1]
            if job.id == job_id:
                policy = RetryPolicy(max_attempts=job.retries or 4)
                success, _ = policy.run(processor, job.data)
                self._jobs.remove(entry)
                return success
        return False

    def get_next_job(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self._jobs:
            return None
        self._jobs.sort(key=lambda x: (x[1].priority, x[0]))
        job = self._jobs[0][1]
        return (job.id, job.data)