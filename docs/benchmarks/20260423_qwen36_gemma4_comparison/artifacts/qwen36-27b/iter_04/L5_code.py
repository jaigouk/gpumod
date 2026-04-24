"""
queue package: Modular job queue implementation with priority and retry support.
"""
from .core import Job, JobQueue
from .priority import PriorityQueue
from .retry import process_with_retry

# Clean public API
__all__ = [
    "Job",
    "JobQueue",
    "PriorityQueue",
    "process_with_retry",
]