import heapq
from typing import Dict, Optional, Tuple, List


class JobQueue:
    def __init__(self) -> None:
        # Each entry in the heap is a tuple:
        # (-priority, FIFO counter, job id, job data)
        self._heap: List[Tuple[int, int, str, Dict]] = []
        self._counter: int = 0

    def add_job(self, job: str, job_data: Dict, priority: int = 0) -> None:
        """
        Add a job to the queue.

        Parameters
        ----------
        job : str
            Identifier for the job (used for identification and ordering within the same priority).
        job_data : dict
            Data associated with the job.
        priority : int, optional
            Prioritization value; default is 0 (normal).
        """
        # Negative priority turns the min‑heap into a max‑heap for our purposes.
        # The FIFO counter ensures that jobs with the same priority retain
        # their original order.
        heapq.heappush(self._heap, (-priority, self._counter, job, job_data))
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """
        Retrieve and return the highest‑priority job available.

        Returns
        -------
        tuple[str, dict] | None
            (job identifier, job data) of the next job to execute, or None if the queue is empty.
        """
        if not self._heap:
            return None
        _, _, job, data = heapq.heappop(self._heap)
        return job, data