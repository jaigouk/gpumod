from collections import deque
import time
from typing import Dict, Optional, Tuple, Deque

class JobQueue:
    """A minimal FIFO job queue.

    Jobs are enqueued with ``add_job`` and their results are retrieved with
    ``get_result``.  The implementation processes jobs sequentially in the order
    they were enqueued.
    """

    def __init__(self) -> None:
        self._queue: Deque[Tuple[str, Dict]] = deque()
        self._processed: Dict[str, Dict] = {}  # maps job_id -> result

    def add_job(self, job_id: str, data: Dict) -> str:
        """Enqueue a new job.

        Parameters
        ----------
        job_id : str
            Unique identifier for the job.
        data : dict
            Payload of the job (will become the result).

        Returns
        -------
        str
            The supplied ``job_id``.

        Raises
        ------
        ValueError
            If the job identifier already exists.
        """
        if job_id in self._processed or job_id in self._queue:
            raise ValueError(f"Job {job_id!r} already exists.")
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict]:
        """Return the result of a job after it has been processed.

        The method sequentially processes jobs from the queue in FIFO order
        until the requested job is completed, then returns its result.
        If the job has already been processed, its result is returned
        immediately. If the job cannot be found, ``None`` is returned.

        Returns
        -------
        dict | None
            The result dictionary for the job, or ``None`` if not found.
        """
        # If the job has already been processed, return its result directly.
        if job_id in self._processed:
            return self._processed[job_id]

        while True:
            # No more jobs to process – the requested job does not exist.
            if not self._queue:
                return None

            # Dequeue the next job in FIFO order.
            current_job_id, current_data = self._queue.popleft()

            # Simulate the execution of the job.
            time.sleep(0.1)  # tiny pause to represent work

            # Record the result (here the payload itself is the result).
            self._processed[current_job_id] = current_data

            # If this is the job we were looking for, return its result.
            if current_job_id == job_id:
                return self._processed[job_id]