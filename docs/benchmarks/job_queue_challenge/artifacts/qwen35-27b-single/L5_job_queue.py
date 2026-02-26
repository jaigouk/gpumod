```python
# queue/__init__.py
"""
Queue package for job processing with retry and priority support.
"""

from .core import JobQueue
from .retry import RetryPolicy
from .priority import PriorityMixin

__all__ = ['JobQueue', 'RetryPolicy', 'PriorityMixin']
```

```python
# queue/core.py
"""
Core JobQueue implementation with basic job processing functionality.
"""

from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import time


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Represents a job to be processed."""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    priority: int = 0  # Default priority for base class


class JobQueue:
    """
    Base job queue implementation with core functionality.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._jobs: List[Job] = []
        self._job_counter = 0
        self._processed_count = 0
        self._failed_count = 0
    
    def add_job(self, func: Callable, *args, **kwargs) -> str:
        """Add a job to the queue."""
        if len(self._jobs) >= self.max_size:
            raise QueueFullError("Queue is full")
        
        self._job_counter += 1
        job_id = f"job_{self._job_counter}"
        
        job = Job(
            id=job_id,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        self._jobs.append(job)
        return job_id
    
    def get_job(self) -> Optional[Job]:
        """Get the next job to process."""
        if not self._jobs:
            return None
        
        # Default: FIFO order
        return self._jobs.pop(0)
    
    def process_job(self, job: Job) -> Any:
        """Process a single job."""
        job.status = JobStatus.RUNNING
        
        try:
            result = job.func(*job.args, **job.kwargs)
            job.result = result
            job.status = JobStatus.COMPLETED
            self._processed_count += 1
            return result
        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
            self._failed_count += 1
            raise
    
    def process_all(self) -> Dict[str, int]:
        """Process all jobs in the queue."""
        results = {"processed": 0, "failed": 0}
        
        while True:
            job = self.get_job()
            if job is None:
                break
            
            try:
                self.process_job(job)
                results["processed"] += 1
            except Exception:
                results["failed"] += 1
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            "total_jobs": len(self._jobs),
            "processed": self._processed_count,
            "failed": self._failed_count,
            "max_size": self.max_size
        }
    
    def clear(self):
        """Clear all jobs from the queue."""
        self._jobs.clear()


class QueueFullError(Exception):
    """Raised when the queue is full."""
    pass
```

```python
# queue/retry.py
"""
Retry policy mixin for job queue with configurable retry behavior.
"""

from typing import Optional, Callable
import time


class RetryPolicy:
    """
    Mixin class that adds retry functionality to JobQueue.
    """
    
    def __init__(self, *args, max_retries: int = 3, 
                 retry_delay: float = 1.0,
                 retry_backoff: float = 2.0,
                 retry_exceptions: tuple = (Exception,),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.retry_exceptions = retry_exceptions
    
    def process_job(self, job):
        """Process a job with retry logic."""
        job.status = JobStatus.RUNNING
        
        while job.retry_count <= self.max_retries:
            try:
                result = job.func(*job.args, **job.kwargs)
                job.result = result
                job.status = JobStatus.COMPLETED
                self._processed_count += 1
                return result
            except self.retry_exceptions as e:
                job.retry_count += 1
                
                if job.retry_count > self.max_retries:
                    job.error = str(e)
                    job.status = JobStatus.FAILED
                    self._failed_count += 1
                    raise
                
                # Calculate delay with exponential backoff
                delay = self.retry_delay * (self.retry_backoff ** (job.retry_count - 1))
                time.sleep(delay)
        
        # Should not reach here, but just in case
        job.status = JobStatus.FAILED
        self._failed_count += 1
        raise Exception("Max retries exceeded")
    
    def add_job(self, func, *args, **kwargs):
        """Add a job with retry support."""
        job_id = super().add_job(func, *args, **kwargs)
        job = self._jobs[-1]
        job.retry_count = 0
        return job_id
```

```python
# queue/priority.py
"""
Priority mixin for job queue with priority-based job ordering.
"""

import heapq
from typing import Optional


class PriorityMixin:
    """
    Mixin class that adds priority-based job ordering to JobQueue.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._priority_heap = []
    
    def add_job(self, func, *args, priority: int = 0, **kwargs):
        """Add a job with priority support."""
        if len(self._priority_heap) >= self.max_size:
            raise QueueFullError("Queue is full")
        
        self._job_counter += 1
        job_id = f"job_{self._job_counter}"
        
        job = Job(
            id=job_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        # Use negative priority for max-heap behavior (higher priority = processed first)
        heapq.heappush(self._priority_heap, (-job.priority, job))
        return job_id
    
    def get_job(self) -> Optional[Job]:
        """Get the highest priority job."""
        if not self._priority_heap:
            return None
        
        _, job = heapq.heappop(self._priority_heap)
        return job
    
    def peek_job(self) -> Optional[Job]:
        """Peek at the highest priority job without removing it."""
        if not self._priority_heap:
            return None
        
        return self._priority_heap[0][1]
    
    def get_jobs_by_priority(self, priority: int) -> list:
        """Get all jobs with a specific priority."""
        return [job for _, job in self._priority_heap if job.priority == priority]
```