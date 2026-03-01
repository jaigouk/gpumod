"""
Queue Package.
A modular job queue implementation.
"""

from .core import Job, JobQueue
from .retry import process_with_retry

# Explicitly define what is exported to the public API
__all__ = [
    "Job",
    "JobQueue",
    "process_with_retry",  # Included for usability, matching original monolithic API
]