```python
# queue/__init__.py
"""Queue package - Job queue with retry and priority support."""

from .core import JobQueue
from .retry import RetryPolicy
from .priority import PriorityMixin

__all__ = ['JobQueue', 'RetryPolicy', 'PriorityMixin']
```

```python
# queue/core.py
"""Core JobQueue class with basic queue functionality."""

from collections import deque
from typing import Any, Callable, Optional, Deque
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Job:
    """Represents a job in the queue."""
    task: Callable
    args: tuple = ()
    kwargs: dict = None
    created_at: datetime = None
    job_id: str = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.job_id is None:
            self.job_id = str(id(self))


class JobQueue:
    """Base job queue with core functionality."""

    def __init__(self, max_size: int = 1000):
        self._queue: Deque[Job] = deque()
        self._max_size = max_size
        self._processed_count = 0
        self._failed_count = 0

    def enqueue(self, task: Callable, *args, **kwargs) -> Job:
        """Add a job to the queue."""
        if len(self._queue) >= self._max_size:
            raise QueueFullError(f"Queue is full (max size: {self._max_size})")

        job = Job(task=task, args=args, kwargs=kwargs)
        self._queue.append(job)
        return job

    def dequeue(self) -> Optional[Job]:
        """Remove and return the next job from the queue."""
        if not self._queue:
            return None
        job = self._queue.popleft()
        return job

    def process(self) -> Optional[Any]:
        """Process the next job in the queue."""
        job = self.dequeue()
        if job is None:
            return None

        try:
            result = job.task(*job.args, **job.kwargs)
            self._processed_count += 1
            return result
        except Exception as e:
            self._failed_count += 1
            raise e

    def size(self) -> int:
        """Return the number of jobs in the queue."""
        return len(self._queue)

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._queue) == 0

    def is_full(self) -> bool:
        """Check if the queue is full."""
        return len(self._queue) >= self._max_size

    def clear(self):
        """Remove all jobs from the queue."""
        self._queue.clear()

    def get_stats(self) -> dict:
        """Return queue statistics."""
        return {
            'size': self.size(),
            'processed': self._processed_count,
            'failed': self._failed_count,
            'max_size': self._max_size
        }


class QueueFullError(Exception):
    """Raised when the queue is full."""
    pass
```

```python
# queue/retry.py
"""Retry policy mixin for job queue."""

from typing import Optional, Callable, Any
from datetime import datetime, timedelta


class RetryPolicy:
    """Mixin class that adds retry functionality to JobQueue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_retries = 3
        self._retry_delay = 1  # seconds
        self._retry_backoff = 2.0  # exponential backoff multiplier
        self._retry_queue = []

    def set_retry_policy(self, max_retries: int = 3, retry_delay: float = 1, 
                         backoff: float = 2.0):
        """Configure retry policy parameters."""
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._retry_backoff = backoff

    def _should_retry(self, job) -> bool:
        """Check if a job should be retried."""
        if not hasattr(job, 'retry_count'):
            job.retry_count = 0
        return job.retry_count < self._max_retries

    def _add_retry(self, job, delay: float):
        """Add a job to the retry queue with delay."""
        job.retry_count = getattr(job, 'retry_count', 0) + 1
        job.retry_at = datetime.now() + timedelta(seconds=delay)
        self._retry_queue.append(job)

    def _process_retries(self):
        """Process jobs that are ready for retry."""
        now = datetime.now()
        ready_jobs = []
        pending_jobs = []

        for job in self._retry_queue:
            if job.retry_at <= now:
                ready_jobs.append(job)
            else:
                pending_jobs.append(job)

        self._retry_queue = pending_jobs

        for job in ready_jobs:
            self.enqueue(job.task, *job.args, **job.kwargs)

    def process_with_retry(self) -> Optional[Any]:
        """Process the next job with retry support."""
        self._process_retries()

        job = self.dequeue()
        if job is None:
            return None

        try:
            result = job.task(*job.args, **job.kwargs)
            self._processed_count += 1
            return result
        except Exception as e:
            self._failed_count += 1

            if self._should_retry(job):
                delay = self._retry_delay * (self._retry_backoff ** getattr(job, 'retry_count', 0))
                self._add_retry(job, delay)
                return None

            raise e

    def get_retry_stats(self) -> dict:
        """Return retry statistics."""
        return {
            'pending_retries': len(self._retry_queue),
            'max_retries': self._max_retries,
            'retry_delay': self._retry_delay,
            'backoff': self._retry_backoff
        }
```

```python
# queue/priority.py
"""Priority mixin for job queue."""

import heapq
from typing import Any, Optional


class PriorityMixin:
    """Mixin class that adds priority queue functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._priority_queue = []
        self._counter = 0

    def enqueue_with_priority(self, task, priority: int = 0, *args, **kwargs):
        """Add a job to the queue with a priority level (lower = higher priority)."""
        if len(self._priority_queue) >= self._max_size:
            raise QueueFullError(f"Priority queue is full (max size: {self._max_size})")

        job = Job(task=task, args=args, kwargs=kwargs)
        job.priority = priority
        self._counter += 1
        heapq.heappush(self._priority_queue, (priority, self._counter, job))
        return job

    def dequeue_with_priority(self) -> Optional[Job]:
        """Remove and return the highest priority job."""
        if not self._priority_queue:
            return None
        _, _, job = heapq.heappop(self._priority_queue)
        return job

    def process_with_priority(self) -> Optional[Any]:
        """Process the highest priority job."""
        job = self.dequeue_with_priority()
        if job is None:
            return None

        try:
            result = job.task(*job.args, **job.kwargs)
            self._processed_count += 1
            return result
        except Exception as e:
            self._failed_count += 1
            raise e

    def get_priority_stats(self) -> dict:
        """Return priority queue statistics."""
        return {
            'size': len(self._priority_queue),
            'is_empty': len(self._priority_queue) == 0
        }

    def size(self) -> int:
        """Override to include priority queue size."""
        return len(self._queue) + len(self._priority_queue)

    def is_empty(self) -> bool:
        """Override to check both queues."""
        return len(self._queue) == 0 and len(self._priority_queue) == 0

    def clear(self):
        """Clear both regular and priority queues."""
        self._queue.clear()
        self._priority_queue.clear()
```