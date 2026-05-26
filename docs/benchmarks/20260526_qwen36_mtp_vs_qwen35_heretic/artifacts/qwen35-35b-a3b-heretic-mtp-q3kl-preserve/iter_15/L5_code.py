"""
Job Queue Package

A clean, modular job queue system with support for priority,
retry logic, and basic queue operations.
"""

from .core import Job, JobQueue
from .priority import PriorityQueue
from .retry import process_with_retry

__all__ = ['Job', 'JobQueue', 'PriorityQueue', 'process_with_retry']
__version__ = '1.0.0'