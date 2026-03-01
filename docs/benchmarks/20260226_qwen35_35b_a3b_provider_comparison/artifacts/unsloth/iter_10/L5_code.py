"""
Job Queue Package

Provides a modular job queue system with priority support and retry logic.
"""

from queue.core import Job, JobQueue
from queue.retry import process_with_retry

__all__ = ["Job", "JobQueue", "process_with_retry"]
__version__ = "1.0.0"