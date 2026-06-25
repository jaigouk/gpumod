from collections import deque
from typing import Dict, Optional, Tuple

class JobQueue:
    def __init__(self) -> None:
        # Each entry is a tuple: (job_id, data, result)
        # result is None until the job is completed
        self._queue: deque[Tuple[str, Dict, Optional[Dict]]] = deque()

    def add_job(self, job_id: str, data: Dict) -> str:
        """Add a job to the end of the queue and return its job_id."""
        self._queue.append((job_id, data, None))
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict]:
        """
        Retrieve the result of the specified job.
        Jobs are processed in FIFO order: any jobs ahead of the requested job
        are removed from the queue (their results are considered retrieved as None).
        Returns the result dict for the job if it exists, otherwise None.
        """
        while self._queue:
            current_job_id, current_data, current_result = self._queue[0]
            if current_job_id == job_id:
                # Simulate processing – here we return the supplied data as the result.
                # In a real implementation the result would be computed by an external thread.
                return current_data
            else:
                # Enforce FIFO order by discarding the front job (its result is considered None)
                self._queue.popleft()
        # Requested job not found; queue is empty.
        return None