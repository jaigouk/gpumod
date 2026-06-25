from typing import Dataclass, Callable, Tuple, Optional, Dict, Any, List
import heapq


@Dataclass
class Job:
    id: str
    data: Dict[str, Any]
    priority: int = 0
    retries: int = 0


class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts
        self.attempt = 1

    def run(self, fn: Callable, data) -> Tuple[bool, int]:
        while self.attempt <= self.max_attempts:
            try:
                fn(data)
                return (True, self.attempt)
            except:
                self.attempt += 1
                # No sleep implemented
        # Exceeded all attempts
        return (False, self.max_attempts)


class JobQueue:
    def __init__(self):
        self.heap: List[Tuple[int, int, str]] = []
        self.heap_counter: int = 0
        self_jobs: Dict[str, Job] = {}

    def add_job(self, job_id: str, data: Dict[str, Any], priority: int = 0) -> None:
        job = Job(job_id, data, priority, 0)
        self.jobs[job_id] = job
        self.heap_counter += 1
        heapq.heappush(self.heap, (-priority, self.heap_counter, job_id))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self.jobs.pop(job_id, None)
        if job is None:
            raise KeyError(f"Job with id {job_id} not found")
        policy = RetryPolicy()
        success, attempts = policy.run(processor, job.data)
        return success

    def get_next_job(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        while self.heap:
            _, _, job_id = heapq.heappop(self.heap)
            job = self.jobs.get(job_id)
            if job is not None:
                del self.jobs[job_id]
                return (job_id, job.data)
            # stale entry, loop continues
        return None