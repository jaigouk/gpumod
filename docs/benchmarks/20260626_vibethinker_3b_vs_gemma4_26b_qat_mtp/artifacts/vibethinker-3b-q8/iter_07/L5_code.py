from typing import dataclass, Callable, Tuple, Dict, Any, List, Optional

@dataclass
class Job:
    id: str
    data: Dict[str, Any]
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable[[Dict[str, Any]], Any], data: Dict[str, Any]) -> Tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                fn(data)
                return (True, attempts + 1)
            except Exception:
                pass
            attempts += 1
        return (False, self.max_attempts)

class JobQueue:
    def __init__(self):
        self._jobs: List[Job] = []

    def add_job(self, job_id: str, data: Dict[str, Any], priority: int = 0) -> None:
        self._jobs.append(Job(job_id, data, priority))

    def process_job(self, job_id: str, processor: Callable[[], Any]) -> bool:
        for i, job in enumerate(self._jobs):
            if job.id == job_id:
                del self._jobs[i]
                success, _ = RetryPolicy().run(lambda d: processor(d), job.data)
                return success
        return False

    def get_next_job(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self._jobs:
            return None
        max_priority = max(job.priority for job in self._jobs)
        for job in self._jobs:
            if job.priority == max_priority:
                return (job.id, job.data)
        return None