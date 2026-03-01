# queue/__init__.py
"""
Job Queue Package.

Provides a priority-based job queue with retry mechanisms.
"""

from .core import Job, JobQueue
from .retry import process_with_retry

# Define the public API for the package
__all__ = [
    "Job",
    "JobQueue",
    "process_with_retry",
]