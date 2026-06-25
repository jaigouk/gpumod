from collections import deque
from typing import Any, Dict, Optional, Tuple

class JobQueue:
    def __init__(self) -> None:
        self._queue: deque[Tuple[str, Dict[str, Any]]] = deque()
        self._results: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> str:
        # Ensure job_id is unique (optional check)
        if any(job_id == cur[0] for cur in self._queue):
            raise ValueError(f"Job id {job_id} already exists")
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        # Process all pending jobs to maintain FIFO order
        while self._queue:
            cur_id, cur_data = self._queue.popleft()
            self._results[cur_id] = self._process(cur_data)

        return self._results.get(job_id)

    @staticmethod
    def _process(data: Dict[str, Any]) -> Dict[str, Any]:
        # Simple identity processing; replace with actual job logic.
        return data