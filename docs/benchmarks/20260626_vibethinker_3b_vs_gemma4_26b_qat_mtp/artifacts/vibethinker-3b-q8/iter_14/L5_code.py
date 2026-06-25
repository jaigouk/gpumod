from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, Union, List
import heapq

@dataclass
class Job:
    id: str
    data: Dict[str, Any]
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts
        self.attempts = 0
        self.backoff_counter = 0

    def run(self, fn: Callable, data: Any) -> Tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            try:
                result = fn(data)
                return (True, attempts)
            except Exception as e:
                # Simulate backoff without sleeping
                self.backoff_counter += 1
                continue
        return (False, self.max_attempts)

class JobQueue:
    def __init__(self):
        self._heap: List[Tuple[int, int, str]] = []
        self._timestamp = 0
        self._jobs: Dict[str, Job] = {}
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: Dict[str, Any], priority: int = 0) -> None:
        job = Job(job_id, data, priority, 0)
        self._jobs[job_id] = job
        timestamp = self._timestamp
        self._timestamp += 1
        heapq.heappush(self._heap, (priority, timestamp, job_id))

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        # Clean up stale entries until we find the desired job
        while self._heap:
            priority, timestamp, current_id = self._heap[0]
            if current_id != job_id:
                heapq.heappop(self._heap)
                continue
            if job_id in self._jobs and self._jobs[job_id].id == job_id:
                break
            heapq.heappop(self._heap)
        else:
            # No such job in the queue
            return False
        # Remove the job from the heap and internal map
        heapq.heappop(self._heap)
        job = self._jobs.pop(job_id)
        # Execute with retry policy
        success, _ = self._retry_policy.run(processor, job.data)
        return success

    def get_next_job(self) -> Union[Tuple[str, Dict[str, Any]], None]:
        while self._heap:
            priority, timestamp, job_id = self._heap[0]
            job = self._jobs.get(job_id)
            if job is not None and job.id == job_id:
                return (job_id, job.data)
            heapq.heappop(self._heap)
        return None