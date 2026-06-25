from collections import deque
from typing import Dict, Optional, Tuple


class JobQueue:
    def __init__(self) -> None:
        # FIFO container for (job_id, data) tuples
        self._queue: deque[Tuple[str, Dict]] = deque()
        # Stores processed results for quick lookup
        self._results: Dict[str, Dict] = {}

    def add_job(self, job_id: str, data: Dict) -> str:
        """Enqueue a new job and return its identifier."""
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict]:
        """Return the result of a completed job, processing earlier jobs as needed."""
        # If the job has already been processed, return its result directly.
        if job_id in self._results:
            return self._results[job_id]

        # Process jobs in FIFO order until we reach the requested job or the queue empties.
        while self._queue and self._queue[0][0] != job_id:
            job_id_cur, data_cur = self._queue.popleft()
            self._results[job_id_cur] = data_cur

        # If the requested job is at the front of the queue, process it and return its result.
        if self._queue and self._queue[0][0] == job_id:
            job_id_cur, data_cur = self._queue.popleft()
            self._results[job_id_cur] = data_cur
            return data_cur

        # Job not found (never added or not yet enqueued).
        return None