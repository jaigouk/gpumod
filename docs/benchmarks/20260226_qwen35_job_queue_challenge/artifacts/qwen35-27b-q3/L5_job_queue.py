```python
# queue/__init__.py
"""Queue package - Job queue with retry and priority support."""

from .core import JobQueue
from .retry import RetryPolicy
from .priority import PriorityMixin

__all__ = ['JobQueue']


class JobQueue(JobQueue, RetryPolicy, PriorityMixin):
    """JobQueue with retry and priority support."""
    pass
```

```python
# queue/core.py
"""Core JobQueue functionality."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Deque
from enum import Enum
import time


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Job:
    """Represents a job in the queue."""
    id: str
    payload: Any
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    retry_count: int = 0


class JobQueue:
    """Base job queue implementation."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize the job queue.
        
        Args:
            max_size: Maximum number of jobs in the queue
        """
        self._queue: Deque[Job] = deque()
        self._max_size = max_size
        self._job_counter = 0
        self._running_jobs: dict[str, Job] = {}
    
    def enqueue(self, payload: Any, job_id: Optional[str] = None) -> str:
        """Add a job to the queue.
        
        Args:
            payload: Job payload/data
            job_id: Optional custom job ID
            
        Returns:
            Job ID string
        """
        if len(self._queue) >= self._max_size:
            raise QueueFullError(f"Queue is full (max size: {self._max_size})")
        
        self._job_counter += 1
        job_id = job_id or f"job_{self._job_counter}"
        
        job = Job(id=job_id, payload=payload)
        self._queue.append(job)
        return job_id
    
    def dequeue(self) -> Optional[Job]:
        """Remove and return the next job from the queue.
        
        Returns:
            Job object or None if queue is empty
        """
        if not self._queue:
            return None
        
        job = self._queue.popleft()
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self._running_jobs[job.id] = job
        return job
    
    def complete(self, job_id: str) -> None:
        """Mark a job as completed.
        
        Args:
            job_id: ID of the job to complete
        """
        job = self._running_jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            del self._running_jobs[job_id]
    
    def fail(self, job_id: str, error: str) -> None:
        """Mark a job as failed.
        
        Args:
            job_id: ID of the job to fail
            error: Error message
        """
        job = self._running_jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error = error
            del self._running_jobs[job_id]
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID.
        
        Args:
            job_id: ID of the job to retrieve
            
        Returns:
            Job object or None if not found
        """
        if job_id in self._running_jobs:
            return self._running_jobs[job_id]
        
        for job in self._queue:
            if job.id == job_id:
                return job
        
        return None
    
    def size(self) -> int:
        """Return the number of jobs in the queue."""
        return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._queue) == 0
    
    def clear(self) -> None:
        """Clear all jobs from the queue."""
        self._queue.clear()
        self._running_jobs.clear()


class QueueFullError(Exception):
    """Raised when the queue is full."""
    pass
```

```python
# queue/retry.py
"""Retry policy mixin for job queue."""

from typing import Optional, Callable
import time


class RetryPolicy:
    """Mixin to add retry functionality to JobQueue."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_retries = 3
        self._retry_delay = 1.0
        self._backoff_multiplier = 2.0
        self._retry_callback: Optional[Callable[[str, int], None]] = None
    
    def set_retry_policy(self, max_retries: int = 3, 
                        retry_delay: float = 1.0,
                        backoff_multiplier: float = 2.0,
                        callback: Optional[Callable[[str, int], None]] = None) -> None:
        """Configure retry policy.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
            backoff_multiplier: Multiplier for exponential backoff
            callback: Optional callback called before each retry
        """
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._backoff_multiplier = backoff_multiplier
        self._retry_callback = callback
    
    def retry(self, job_id: str) -> bool:
        """Retry a failed job.
        
        Args:
            job_id: ID of the job to retry
            
        Returns:
            True if retry was successful, False otherwise
        """
        job = self.get_job(job_id)
        if not job or job.status.value != "failed":
            return False
        
        if job.retry_count >= self._max_retries:
            return False
        
        job.retry_count += 1
        job.status = JobStatus.RETRYING
        
        if self._retry_callback:
            self._retry_callback(job_id, job.retry_count)
        
        delay = self._retry_delay * (self._backoff_multiplier ** job.retry_count)
        time.sleep(delay)
        
        job.status = JobStatus.PENDING
        job.started_at = None
        job.completed_at = None
        job.error = None
        
        self.enqueue(job.payload, job_id)
        return True
    
    def should_retry(self, job_id: str) -> bool:
        """Check if a job should be retried.
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            True if job can be retried, False otherwise
        """
        job = self.get_job(job_id)
        if not job or job.status.value != "failed":
            return False
        
        return job.retry_count < self._max_retries
```

```python
# queue/priority.py
"""Priority mixin for job queue."""

from typing import Any, Optional
from collections import deque
import heapq


class PriorityMixin:
    """Mixin to add priority queue functionality."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._priority_queue: list[tuple[int, int, Any]] = []
        self._priority_counter = 0
    
    def enqueue_priority(self, payload: Any, 
                        priority: int = 0,
                        job_id: Optional[str] = None) -> str:
        """Add a job with priority to the queue.
        
        Args:
            payload: Job payload/data
            priority: Priority level (lower = higher priority)
            job_id: Optional custom job ID
            
        Returns:
            Job ID string
        """
        self._priority_counter += 1
        job_id = job_id or f"job_{self._priority_counter}"
        
        job = Job(id=job_id, payload=payload)
        heapq.heappush(self._priority_queue, (priority, self._priority_counter, job))
        return job_id
    
    def dequeue(self) -> Optional[Job]:
        """Remove and return the highest priority job.
        
        Returns:
            Job object or None if queue is empty
        """
        if self._priority_queue:
            _, _, job = heapq.heappop(self._priority_queue)
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self._running_jobs[job.id] = job
            return job
        
        return super().dequeue()
    
    def get_priority(self, job_id: str) -> Optional[int]:
        """Get the priority of a job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Priority level or None if not found
        """
        for priority, _, job in self._priority_queue:
            if job.id == job_id:
                return priority
        return None
    
    def size(self) -> int:
        """Return the total number of jobs (priority + regular)."""
        return len(self._priority_queue) + super().size()
    
    def clear(self) -> None:
        """Clear all jobs from the queue."""
        self._priority_queue.clear()
        super().clear()
```