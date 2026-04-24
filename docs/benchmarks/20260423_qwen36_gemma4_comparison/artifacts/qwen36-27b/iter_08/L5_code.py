"""
Queue Package
A modular job queue implementation with priority support and retry logic.
"""

from .core import Job, JobQueue
from .priority import PriorityQueue
from .retry import process_with_retry

# Define the public interface
__all__ = [
    "Job",
    "JobQueue",
    "PriorityQueue",
    "process_with_retry",
]