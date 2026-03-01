"""
Job Queue Package

Provides a comprehensive job queue system with priority support,
retry logic, and exponential backoff.
"""

from .core import Job, JobQueue
from .priority import PriorityQueue
from .retry import process_with_retry

__all__ = [
    "Job",
    "JobQueue",
    "PriorityQueue",
    "process_with_retry",
    "__version__",
]

__version__ = "1.0.0"