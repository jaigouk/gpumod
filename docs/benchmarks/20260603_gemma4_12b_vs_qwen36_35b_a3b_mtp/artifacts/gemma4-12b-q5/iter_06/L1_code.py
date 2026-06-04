from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Internal storage for jobs to maintain FIFO order
        self.queue: deque[tuple[str, dict]] = deque()
        # Storage for completed job results
        self.results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.
        """
        self.queue.append((job_id, data))
        return job_id

    def complete_job(self, job_id: str, result: dict) -> None:
        """
        Helper method to mark a job as completed and store its result.
         This allows the 'After processing...' step in the usage example to function.
        """
        self.results[job_id] = result
        # Remove from internal queue once processed
        try:
            # Note: Removing a specific item from a deque is O(n)
            # In a production system, one might use a dictionary to track 
            # queue positions or a different data structure.
            for i, (jid, _) in enumerate(self.queue):
                if jid == job_id:
                    del self.queue[i]
                    break
        except IndexError:
            pass

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Returns the result of a completed job, or None if not yet completed.
        """
        return self.results.get(job_id)